// This file is part of Substrate.

// Copyright (C) Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: GPL-3.0-or-later WITH Classpath-exception-2.0

//! Native RISC-V executor for PolkaVM PVF blobs.
//!
//! Translates PVM bytecode to native RV64IMAC + xtheadbb + xtheadcondmov.
//! Designed for T-Head C910/TH1520 (Lichee Pi 4A).
//!
//! # Architecture
//!
//! - [`riscv_emit`]: Low-level RISC-V instruction encoder
//! - [`translator`]: PVM→RISC-V bytecode translator
//! - [`memory`]: Guest memory and executable code buffer
//! - [`trampoline`]: Host function call dispatch

mod memory;
mod riscv_emit;
mod trampoline;
mod translator;

use memory::{ExecutableBuffer, GuestMemory};
use polkavm_common::abi::{MemoryMapBuilder, VM_MAX_PAGE_SIZE};
use sc_executor_common::{
	error::{Error, WasmError},
	wasm_runtime::{AllocationStats, WasmInstance, WasmModule},
};
use sp_runtime_interface::unpack_ptr_and_len;
use sp_wasm_interface::{Function, HostFunctions};
use std::sync::Arc;
use trampoline::HostContext;
use translator::Translator;

const GUEST_ADDR_SPACE: usize = 0x1_0000_0000;
const LOG_TARGET: &'static str = "executor-native-riscv";

/// Shared data between module and instances
struct SharedModuleData {
	/// Native machine code (RX mmap).
	code: ExecutableBuffer,
	/// PVM PC → native byte offset mapping. Used for exports.
	pc_to_native: Vec<u32>,
	/// PVM address → native byte offset mapping. Used for indirect jumps.
	/// Indexed by PVM address (aligned to VM_CODE_ADDRESS_ALIGNMENT=2).
	/// Built from blob.jump_table().
	addr_to_native: Vec<u32>,
	/// Export name → PVM PC.
	exports: Vec<(String, u32)>,
	/// Import index → host function reference (`None` = unresolved / grow_heap).
	host_functions: Vec<Option<&'static dyn Function>>,
	/// Import index of `grow_heap`, if present.
	grow_heap_index: Option<usize>,
	/// Heap base address (guest address).
	heap_base: u32,
	/// Maximum heap size.
	max_heap_size: u32,
	/// Initial RO data to write into guest memory.
	ro_data: Vec<u8>,
	/// Initial RW data to write into guest memory.
	rw_data: Vec<u8>,
	/// Stack size.
	stack_size: u32,
	/// RO data section size.
	ro_data_size: u32,
	/// RW data section size.
	rw_data_size: u32,
}

// SAFETY: ExecutableBuffer's raw pointers are only dereferenced single-threaded.
unsafe impl Send for SharedModuleData {}
unsafe impl Sync for SharedModuleData {}

/// Compiled native RISC-V module
pub struct NativeRiscvModule {
	shared: Arc<SharedModuleData>,
}

impl WasmModule for NativeRiscvModule {
	fn new_instance(&self) -> Result<Box<dyn WasmInstance>, Error> {
		// Allocate guest address space
		let mut guest_memory = GuestMemory::new(GUEST_ADDR_SPACE)?;

		let memory_map = MemoryMapBuilder::new(VM_MAX_PAGE_SIZE)
			.ro_data_size(self.shared.ro_data_size)
			.rw_data_size(self.shared.rw_data_size)
			.stack_size(self.shared.stack_size)
			.build()
			.map_err(|e| Error::from(format!("failed to build memory map: {e}")))?;

		if !self.shared.ro_data.is_empty() {
			guest_memory.write(memory_map.ro_data_address(), &self.shared.ro_data)?;
		}

		if !self.shared.rw_data.is_empty() {
			guest_memory.write(memory_map.rw_data_address(), &self.shared.rw_data)?;
		}

		Ok(Box::new(NativeRiscvInstance {
			shared: Arc::clone(&self.shared),
			guest_memory,
			memory_map_stack_high: memory_map.stack_address_high(),
			memory_map_rw_data_addr: memory_map.rw_data_address(),
		}))
	}
}

/// Instance of native RISC-V execution
pub struct NativeRiscvInstance {
	shared: Arc<SharedModuleData>,
	guest_memory: GuestMemory,
	memory_map_stack_high: u32,
	memory_map_rw_data_addr: u32,
}

impl WasmInstance for NativeRiscvInstance {
	fn call_with_allocation_stats(
		&mut self,
		name: &str,
		raw_data: &[u8],
	) -> (Result<Vec<u8>, Error>, Option<AllocationStats>) {
		let result = self.call_inner(name, raw_data);
		(result, None)
	}
}

impl NativeRiscvInstance {
	fn call_inner(&mut self, name: &str, raw_data: &[u8]) -> Result<Vec<u8>, Error> {
		// Find the export
		let pvm_pc = self
			.shared
			.exports
			.iter()
			.find(|(n, _)| n == name)
			.map(|(_, pc)| *pc)
			.ok_or_else(|| {
				Error::from(format!("cannot call into the runtime: export not found: '{name}'"))
			})?;

		let Ok(raw_data_length) = u32::try_from(raw_data.len()) else {
			return Err(
				format!("cannot call runtime method '{name}': input payload is too big").into(),
			);
		};

		log::info!(
			target: LOG_TARGET,
			"call_inner '{name}' pvm_pc={pvm_pc} raw_data_len={raw_data_length}"
		);

		// Reset RW memory (re-write initial data)
		if !self.shared.rw_data.is_empty() {
			self.guest_memory
				.write(self.memory_map_rw_data_addr, &self.shared.rw_data)?;
		}

		// Look up native offset
		let native_offset = self
			.shared
			.pc_to_native
			.get(pvm_pc as usize)
			.copied()
			.unwrap_or(u32::MAX);
		if native_offset == u32::MAX {
			return Err(format!(
				"cannot call runtime method '{name}': PVM PC {pvm_pc} has no native code"
			)
			.into());
		}

		let entry_point =
			unsafe { self.shared.code.base_ptr().add(native_offset as usize) } as *const u8;

		// Set up host context for ecalli dispatch
		let mut host_ctx = HostContext {
			memory: &mut self.guest_memory as *mut GuestMemory,
			host_functions: self.shared.host_functions.clone(),
			grow_heap_index: self.shared.grow_heap_index,
			heap_top: self.shared.heap_base,
			heap_base: self.shared.heap_base,
			max_heap_size: self.shared.max_heap_size,
			input_data: Some(raw_data.to_vec()),
			last_error: None,
			call_counter: 0,
		};

		let mem_base = self.guest_memory.base_ptr() as u64;
		// sp must hold the GUEST address (32-bit) — the translated code adds
		// mem_base via guest_addr_to_host for all memory accesses.
		let stack_ptr = self.memory_map_stack_high as u64;
		let ctx_ptr = &mut host_ctx as *mut HostContext as u64;
		let code_base = self.shared.code.base_ptr() as u64;
		let addr_to_native_ptr = self.shared.addr_to_native.as_ptr() as u64;

		log::trace!(
			target: LOG_TARGET,
			"about to call_native: entry_point={entry_point:?} \
			 native_offset=0x{native_offset:x} mem_base=0x{mem_base:x} \
			 stack_ptr=0x{stack_ptr:x} ctx_ptr=0x{ctx_ptr:x} arg={raw_data_length} \
			 code_base=0x{code_base:x} addr_to_native=0x{addr_to_native_ptr:x}"
		);

		// SAFETY: entry_point is valid executable code from our translation,
		// all pointers are valid for the duration of this call.
		let result_packed: u64 = unsafe {
			call_native(
				mem_base,
				ctx_ptr,
				stack_ptr,
				raw_data_length as u64,
				entry_point as u64,
				code_base,
				addr_to_native_ptr,
			)
		};

		log::debug!(target: LOG_TARGET, "result_packed=0x{result_packed:016x}");

		// Check for host call errors
		if let Some(error) = host_ctx.last_error {
			return Err(
				format!("call into the runtime method '{name}' failed: {error}").into(),
			);
		}

		// Unpack result pointer+length from packed u64 in a0
		let (result_pointer, result_length) = unpack_ptr_and_len(result_packed);
		log::info!(
			target: LOG_TARGET,
			"result pointer=0x{result_pointer:x} length={result_length}"
		);
		let output = self.guest_memory.read(result_pointer, result_length)?;

		Ok(output)
	}
}

/// Create a native RISC-V runtime from a PVM blob.
pub fn create_runtime<H>(
	blob: &polkavm_common::program::ProgramBlob,
) -> Result<Box<dyn WasmModule>, WasmError>
where
	H: HostFunctions,
{
	// Collect host functions and build import index mapping.
	let host_fns = H::host_functions();
	let host_fn_map: std::collections::HashMap<&str, &'static dyn Function> =
		host_fns.iter().map(|f| (f.name(), *f)).collect();

	// Map import names to host function table indices.
	// `grow_heap` is handled specially by the dispatch.
	// Unresolved imports are stored as `None` and will trap if called.
	let imports = blob.imports();
	let mut host_functions_table: Vec<Option<&'static dyn Function>> = Vec::new();
	let mut grow_heap_index: Option<usize> = None;
	for maybe_symbol in imports.iter() {
		let idx = host_functions_table.len();
		if let Some(symbol) = maybe_symbol {
			let raw_bytes = symbol.as_bytes();
			// Strip trailing null bytes if present (PVM blob format may include them)
			let trimmed = match raw_bytes.iter().position(|&b| b == 0) {
				Some(pos) => &raw_bytes[..pos],
				None => raw_bytes,
			};
			let name = std::str::from_utf8(trimmed).unwrap_or("");
			if let Some(&hf) = host_fn_map.get(name) {
				host_functions_table.push(Some(hf));
			} else if name == "grow_heap" {
				grow_heap_index = Some(idx);
				host_functions_table.push(None); // handled by dispatch
			} else {
				host_functions_table.push(None);
			}
		} else {
			host_functions_table.push(None);
		}
	}

	// --- Translate PVM bytecode to native RISC-V ---

	// Collect all instructions and find max PC
	let mut max_pc: u32 = 0;
	let mut instruction_list = Vec::new();
	for parsed in blob.instructions() {
		if parsed.offset.0 > max_pc {
			max_pc = parsed.offset.0;
		}
		if parsed.next_offset.0 > max_pc {
			max_pc = parsed.next_offset.0;
		}
		instruction_list.push(parsed);
	}

	// Compute addr_to_native table size from jump table before translation,
	// so the translator can embed the correct bounds check in indirect jumps.
	let jump_table = blob.jump_table();
	let addr_table_len = if jump_table.is_empty() {
		1 // At least 1 entry (index 0 = invalid PVM address)
	} else {
		// Max valid PVM address = jump_table.len() * VM_CODE_ADDRESS_ALIGNMENT
		// Table size = max_address + 1
		(jump_table.len() as usize) * 2 + 1
	};

	let mut translator = Translator::new(max_pc);
	translator.set_addr_table_len(addr_table_len);

	// Generate the trampoline at the beginning of the code buffer.
	// ecalli instructions will `jal` to this offset.
	let dispatch_fn_addr = trampoline::dispatch_host_call as *const () as u64;
	let trampoline_code = trampoline::generate_trampoline(dispatch_fn_addr);

	let trampoline_offset = 0;
	for chunk in trampoline_code.chunks_exact(4) {
		let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
		translator.asm.emit32_pub(word);
	}

	translator.set_trampoline_offset(trampoline_offset);

	// Generate a return stub for PVM function returns (jump to PVM address 0).
	// When the top-level function returns, RA=0 (set by call_native). The indirect
	// jump looks up addr_to_native[0] and finds this stub, which restores the host
	// stack and jumps to the saved host return address in s7 (x23).
	let return_stub_offset = translator.asm.position() as u32;
	translator.asm.mv(2, 19);       // sp = s3 (restore host stack pointer)
	translator.asm.jalr(0, 23, 0);  // jalr x0, s7, 0 (jump to host return addr, no link)

	// Translate each PVM instruction
	for parsed in &instruction_list {
		translator.set_current_pc(parsed.offset.0);
		parsed.kind.visit(&mut translator);
		// Map `next_offset` to current native position so that return addresses
		// (set by `load_imm_and_jump`) resolve correctly in indirect jumps.
		translator.map_pc(parsed.next_offset.0);
	}

	// Resolve all branch/jump fixups
	translator.resolve_fixups();

	let (native_code, pc_to_native) = translator.into_parts();

	// Build `addr_to_native` table from `blob.jump_table()`.
	// PVM indirect jumps use PVM addresses (aligned to 2), not byte offsets.
	// The jump table maps: index → byte_offset (ProgramCounter).
	// PVM address = (index + 1) * VM_CODE_ADDRESS_ALIGNMENT.
	// We build a table indexed by PVM address → native offset.
	let mut addr_to_native = vec![u32::MAX; addr_table_len];
	// Map PVM address 0 to the return stub so that top-level function returns
	// (`jump_indirect(RA=0, 0)`) route to the return stub instead of trapping.
	addr_to_native[0] = return_stub_offset;
	let mut addr_mapped = 0u32;
	for (index, code_offset) in jump_table.iter().enumerate() {
		let pvm_addr = ((index as u32) + 1) * 2; // VM_CODE_ADDRESS_ALIGNMENT = 2
		let byte_offset = code_offset.0 as usize;
		if let Some(&native_off) = pc_to_native.get(byte_offset) {
			if native_off != u32::MAX && (pvm_addr as usize) < addr_to_native.len() {
				addr_to_native[pvm_addr as usize] = native_off;
				addr_mapped += 1;
			}
		}
	}

	let exports: Vec<(String, u32)> = blob
		.exports()
		.filter_map(|export| {
			let name = std::str::from_utf8(export.symbol().as_bytes())
				.ok()?
				.to_string();
			Some((name, export.program_counter().0))
		})
		.collect();

	let mapped_count = pc_to_native.iter().filter(|&&v| v != u32::MAX).count();
	log::debug!(
		target: LOG_TARGET,
		"compiled {} PVM instructions to {} bytes of native code, {} exports, \
		 pc_to_native: {}/{} mapped, addr_to_native: {}/{} mapped (jump_table: {} entries)",
		instruction_list.len(),
		native_code.len(),
		exports.len(),
		mapped_count,
		pc_to_native.len(),
		addr_mapped,
		addr_to_native.len(),
		jump_table.len(),
	);

	let memory_map = MemoryMapBuilder::new(VM_MAX_PAGE_SIZE)
		.ro_data_size(blob.ro_data_size())
		.rw_data_size(blob.rw_data_size())
		.stack_size(blob.stack_size())
		.build()
		.map_err(|e| WasmError::Other(format!("failed to build memory map: {e}")))?;

	log::debug!(
		target: LOG_TARGET,
		"heap_base=0x{:x} max_heap_size=0x{:x} grow_heap_index={:?}",
		memory_map.heap_base(),
		memory_map.max_heap_size(),
		grow_heap_index,
	);

	// Build executable buffer
	let code = ExecutableBuffer::new(&native_code)
		.map_err(|e| WasmError::Other(e.to_string()))?;

	Ok(Box::new(NativeRiscvModule {
		shared: Arc::new(SharedModuleData {
			code,
			pc_to_native,
			addr_to_native,
			exports,
			host_functions: host_functions_table,
			grow_heap_index,
			heap_base: memory_map.heap_base(),
			max_heap_size: memory_map.max_heap_size(),
			ro_data: blob.ro_data().to_vec(),
			rw_data: blob.rw_data().to_vec(),
			stack_size: blob.stack_size(),
			ro_data_size: blob.ro_data_size(),
			rw_data_size: blob.rw_data_size(),
		}),
	}))
}

/// Call native RISC-V code, switching to the guest stack and setting up
/// context registers. Returns the packed result from a0.
///
/// # Safety
///
/// `entry` must point to valid executable RISC-V code. Other pointers must
/// be valid for the duration of the call.
#[cfg(target_arch = "riscv64")]
unsafe fn call_native(
	mem_base: u64,
	ctx_ptr: u64,
	stack_ptr: u64,
	arg: u64,
	entry: u64,
	code_base: u64,
	addr_to_native_ptr: u64,
) -> u64 {
	let result: u64;
	// We manually save/restore callee-saved registers, gp, tp inside the asm
	// block. We use clobber_abi("C") to tell the compiler that all caller-saved
	// registers (a7, t0-t6, ra) are trashed by the guest code. Inputs are bound
	// to specific a-registers (a0-a6) so the compiler won't allocate them to
	// callee-saved registers that our asm overwrites (s2, s4, s5, etc.).
	//
	// The return mechanism: PVM convention is RA=0 for the top-level call.
	// When the function returns via jump_indirect(RA=0, 0), the native code
	// looks up addr_to_native[0] which points to a return stub. The stub
	// restores sp from s3 and jumps to s7 (the saved host return address).
	core::arch::asm!(
		// Save callee-saved host registers + ra/gp/tp
		"addi sp, sp, -128",
		"sd ra,   0(sp)",
		"sd s0,   8(sp)",
		"sd s1,  16(sp)",
		"sd s2,  24(sp)",
		"sd s3,  32(sp)",
		"sd s4,  40(sp)",
		"sd s5,  48(sp)",
		"sd s6,  56(sp)",
		"sd s7,  64(sp)",
		"sd s8,  72(sp)",
		"sd s9,  80(sp)",
		"sd s10, 88(sp)",
		"sd s11, 96(sp)",
		"sd gp, 104(sp)",
		"sd tp, 112(sp)",
		// Save host sp in s3 (guest doesn't use s3+)
		"mv s3, sp",
		// Set up guest context registers from a-register inputs.
		// a0 already has the argument; a4 holds entry (read by jalr below).
		"mv s2, a1",                  // s2 = guest memory base
		"mv gp, a2",                  // gp = HostContext pointer
		"mv s4, a5",                  // s4 = native code base (for indirect jumps)
		"mv s5, a6",                  // s5 = addr_to_native table (for indirect jumps)
		"mv sp, a3",                  // sp = guest stack (32-bit address)
		// Save host return address in s7 for the return stub.
		// Disable C-extension compression so all instructions are exactly
		// 4 bytes and the hardcoded offset is correct.
		".option push",
		".option norvc",
		// auipc s7, 0 loads PC of this instruction into s7.
		// addi s7, s7, 16 advances past: addi(4) + mv(4) + jalr(4) = 12 bytes
		// plus the addi itself (4 bytes) = 16 total to reach "mv s6, a0".
		"auipc s7, 0",
		"addi s7, s7, 16",
		// Set RA=0 (PVM convention: top-level call has RA=0)
		"mv ra, zero",
		// Jump to translated code (no link — return via return stub + s7)
		"jalr zero, a4, 0",
		".option pop",
		// Return point: the return stub jumps here via s7.
		// Stash result (a0) into s6 before we restore host state.
		"mv s6, a0",
		// Restore host sp (s3 was updated by return stub)
		"mv sp, s3",
		// Restore callee-saved host registers
		"ld ra,   0(sp)",
		"ld s0,   8(sp)",
		"ld s1,  16(sp)",
		"ld s2,  24(sp)",
		"ld s3,  32(sp)",
		"ld s4,  40(sp)",
		"ld s5,  48(sp)",
		// s6 restore skipped — it holds our result
		"ld s7,  64(sp)",
		"ld s8,  72(sp)",
		"ld s9,  80(sp)",
		"ld s10, 88(sp)",
		"ld s11, 96(sp)",
		"ld gp, 104(sp)",
		"ld tp, 112(sp)",
		// Move result from s6 to a0 (output), then restore s6
		"mv a0, s6",
		"ld s6,  56(sp)",
		"addi sp, sp, 128",
		// Inputs bound to specific a-registers to avoid conflicts with
		// callee-saved registers that our asm temporarily overwrites.
		inout("a0") arg => result,    // a0: argument in, result out
		in("a1") mem_base,
		in("a2") ctx_ptr,
		in("a3") stack_ptr,
		in("a4") entry,
		in("a5") code_base,
		in("a6") addr_to_native_ptr,
		// Guest code clobbers all caller-saved registers. a0-a6 are excluded
		// from the clobber set since they are explicit operands above.
		// Remaining clobbers: a7, t0-t6, ra.
		clobber_abi("C"),
	);
	result
}

#[cfg(not(target_arch = "riscv64"))]
unsafe fn call_native(
	_mem_base: u64,
	_ctx_ptr: u64,
	_stack_ptr: u64,
	_arg: u64,
	_entry: u64,
	_code_base: u64,
	_addr_to_native_ptr: u64,
) -> u64 {
	panic!("native RISC-V execution is only supported on riscv64 targets")
}

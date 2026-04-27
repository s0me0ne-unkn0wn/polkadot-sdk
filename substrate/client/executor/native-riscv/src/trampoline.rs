// This file is part of Substrate.

// Copyright (C) Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: GPL-3.0-or-later WITH Classpath-exception-2.0

//! Host function trampoline for native RISC-V execution.
//!
//! When guest code calls `ecalli N`, the translator emits:
//!   li a7, N     # host function index
//!   jal ra, trampoline
//!
//! The trampoline saves guest registers, switches to the host stack,
//! calls a Rust dispatch function, restores guest registers and returns.

use crate::memory::GuestMemory;
use crate::riscv_emit::Emitter;
use sp_wasm_interface::{Function, FunctionContext, Pointer, Value, ValueType, WordSize};

// We save/restore all guest registers (RA, SP, T0-T2, S0-S1, A0-A5) plus A7 (index).
// The host dispatch function receives the host function index in `a7` and can
// read/write argument registers (a0-a5) through the saved frame.
//
// The ecalli instruction links to TMP2 (t4/x29) instead of RA to avoid
// clobbering the guest's RA register. The trampoline saves TMP2 as the
// return address separately.
//
// Stack frame layout (offsets from SP after push):
//   0:  RA  (x1, guest return address — NOT clobbered by ecalli)
//   8:  SP  (x2, guest)
//  16:  T0  (x5)
//  24:  T1  (x6)
//  32:  T2  (x7)
//  40:  S0  (x8)
//  48:  S1  (x9)
//  56:  A0  (x10)
//  64:  A1  (x11)
//  72:  A2  (x12)
//  80:  A3  (x13)
//  88:  A4  (x14)
//  96:  A5  (x15)
// 104:  A7  (x17, import index)
// 112:  s2  (x18, mem_base — preserve)
// 120:  gp  (x3, ctx pointer — preserve)
// 128:  return address (from TMP2/t4, set by ecalli's jalr)

const FRAME_SIZE: i32 = 144; // 18 slots * 8 bytes, aligned to 16

/// Generate the trampoline machine code that the translator can `jalr` into.
///
/// `dispatch_addr` is the host address of the `dispatch_host_call` function.
/// Returns the generated trampoline code bytes.
///
/// # Register convention on entry
///
/// - sp (x2)  = guest stack pointer (32-bit guest address, NOT a host address)
/// - s2 (x18) = guest memory base pointer (host address)
/// - s3 (x19) = saved host stack pointer
/// - gp (x3)  = pointer to HostContext
/// - a7 (x17) = import index (from ecalli)
/// - t4 (x29) = return address (from ecalli's `jalr t4, ...`)
/// - ra (x1)  = guest RA (NOT clobbered by ecalli — ecalli links to t4 instead)
///
/// The trampoline switches to the host stack (s3) before saving registers
/// and calling the dispatch function, then switches back to the guest sp
/// before returning.
pub fn generate_trampoline(dispatch_addr: u64) -> Vec<u8> {
	let mut asm = Emitter::new();

	// === Switch to host stack and save guest registers ===
	// Save guest sp in t3, then switch sp to host stack (s3)
	asm.mv(28, 2); // t3 = guest sp
	asm.mv(2, 19); // sp = s3 (host stack)

	// Allocate frame on host stack
	asm.addi(2, 2, -FRAME_SIZE); // sp -= FRAME_SIZE

	// Save guest registers to host stack frame
	asm.sd(1, 2, 0); // save guest ra (original, not clobbered by ecalli)
	asm.sd(28, 2, 8); // save guest sp (from t3)
	asm.sd(5, 2, 16); // save t0
	asm.sd(6, 2, 24); // save t1
	asm.sd(7, 2, 32); // save t2
	asm.sd(8, 2, 40); // save s0
	asm.sd(9, 2, 48); // save s1
	asm.sd(10, 2, 56); // save a0
	asm.sd(11, 2, 64); // save a1
	asm.sd(12, 2, 72); // save a2
	asm.sd(13, 2, 80); // save a3
	asm.sd(14, 2, 88); // save a4
	asm.sd(15, 2, 96); // save a5
	asm.sd(17, 2, 104); // save a7 (import index)
	asm.sd(18, 2, 112); // save s2 (mem_base)
	asm.sd(3, 2, 120); // save gp (ctx)
	asm.sd(29, 2, 128); // save return address (from TMP2/t4)

	// === Call dispatch ===
	// a0 = pointer to HostContext (from gp/x3)
	// a1 = pointer to saved frame (current sp on host stack)
	// a2 = import index (from a7/x17)
	asm.mv(10, 3); // a0 = gp (HostContext*)
	asm.mv(11, 2); // a1 = sp (saved frame on host stack)
	asm.mv(12, 17); // a2 = a7 (import index)

	// Load dispatch address into t3 and call it
	asm.li64(28, dispatch_addr as i64); // t3 = dispatch_addr
	asm.jalr(1, 28, 0); // call dispatch; ra = Rust return

	// === Restore guest registers ===
	asm.ld(1, 2, 0); // restore guest ra (original value)
	// guest sp restored below (after we're done with host sp)
	asm.ld(5, 2, 16); // restore t0
	asm.ld(6, 2, 24); // restore t1
	asm.ld(7, 2, 32); // restore t2
	asm.ld(8, 2, 40); // restore s0
	asm.ld(9, 2, 48); // restore s1
	asm.ld(10, 2, 56); // restore a0 (may have been updated by dispatch)
	asm.ld(11, 2, 64); // restore a1
	asm.ld(12, 2, 72); // restore a2
	asm.ld(13, 2, 80); // restore a3
	asm.ld(14, 2, 88); // restore a4
	asm.ld(15, 2, 96); // restore a5
	asm.ld(18, 2, 112); // restore s2 (mem_base)
	asm.ld(3, 2, 120); // restore gp (ctx)

	// Load saved guest sp and return address
	asm.ld(28, 2, 8); // t3 = saved guest sp
	asm.ld(29, 2, 128); // t4 = return address

	// Deallocate host frame, restore s3, switch back to guest sp
	asm.addi(2, 2, FRAME_SIZE); // sp += FRAME_SIZE (host stack restored)
	asm.mv(19, 2); // s3 = sp (update saved host sp)
	asm.mv(2, 28); // sp = t3 (restore guest sp)
	asm.jalr(0, 29, 0); // return to ecalli caller via t4 (no link)

	asm.into_code()
}

/// Saved register frame for accessing guest registers during host calls.
#[repr(C)]
pub struct SavedFrame {
	pub ra: u64,
	pub sp: u64,
	pub t0: u64,
	pub t1: u64,
	pub t2: u64,
	pub s0: u64,
	pub s1: u64,
	pub a0: u64,
	pub a1: u64,
	pub a2: u64,
	pub a3: u64,
	pub a4: u64,
	pub a5: u64,
	pub a7: u64,
	pub s2: u64,
	pub gp: u64,
	pub return_addr: u64,
}

/// Host context passed to the dispatch function.
/// Contains everything needed to service host calls.
pub struct HostContext {
	/// The guest memory region.
	pub memory: *mut GuestMemory,
	/// Host functions table (indexed by import number).
	/// `None` entries are unresolved imports that should trap.
	pub host_functions: Vec<Option<&'static dyn Function>>,
	/// Import index of `grow_heap`, if present.
	pub grow_heap_index: Option<usize>,
	/// Current heap top (guest address). Starts at heap_base.
	pub heap_top: u32,
	/// Heap base address (guest address, from memory map).
	pub heap_base: u32,
	/// Maximum heap size (from memory map).
	pub max_heap_size: u32,
	/// Input data for the runtime call.
	pub input_data: Option<Vec<u8>>,
	/// Error message from the last host call, if any.
	pub last_error: Option<String>,
	/// Call counter for comparison-mode debugging.
	pub call_counter: u32,
}

// SAFETY: HostContext is used single-threaded within one execution.
unsafe impl Send for HostContext {}
unsafe impl Sync for HostContext {}

/// FunctionContext implementation that reads/writes through guest memory.
struct NativeFunctionContext<'a> {
	memory: &'a mut GuestMemory,
	input_data: &'a mut Option<Vec<u8>>,
	#[allow(dead_code)]
	frame: &'a mut SavedFrame,
}

impl<'a> FunctionContext for NativeFunctionContext<'a> {
	fn read_memory_into(
		&mut self,
		address: Pointer<u8>,
		dest: &mut [u8],
	) -> sp_wasm_interface::Result<()> {
		let data = self
			.memory
			.read(u32::from(address), dest.len() as u32)
			.map_err(|e| e.to_string())?;
		dest.copy_from_slice(&data);
		Ok(())
	}

	fn write_memory(&mut self, address: Pointer<u8>, data: &[u8]) -> sp_wasm_interface::Result<()> {
		self.memory.write(u32::from(address), data).map_err(|e| e.to_string())
	}

	fn allocate_memory(&mut self, _size: WordSize) -> sp_wasm_interface::Result<Pointer<u8>> {
		unimplemented!()
	}

	fn deallocate_memory(&mut self, _ptr: Pointer<u8>) -> sp_wasm_interface::Result<()> {
		unimplemented!()
	}

	fn register_panic_error_message(&mut self, _message: &str) {
		unimplemented!()
	}

	fn take_input_data(&mut self) -> sp_wasm_interface::Result<Vec<u8>> {
		self.input_data
			.take()
			.ok_or_else(|| "Input data already taken".into())
	}

	fn virtualization(&mut self) -> &mut dyn sp_wasm_interface::Virtualization {
		todo!("Implement virtualization for native RISC-V")
	}
}

/// Dispatch a host function call. Called from the trampoline via function pointer.
///
/// # Safety
///
/// `ctx_ptr` must point to a valid `HostContext`, and `frame_ptr` must point to
/// a valid `SavedFrame` on the stack. Called only from generated trampoline code.
pub unsafe extern "C" fn dispatch_host_call(
	ctx_ptr: *mut HostContext,
	frame_ptr: *mut SavedFrame,
	import_index: u64,
) {
	let ctx = &mut *ctx_ptr;
	let frame = &mut *frame_ptr;

	let index = import_index as usize;
	ctx.call_counter += 1;
	log::trace!(
		target: crate::LOG_TARGET,
		"dispatch_host_call index={index} total={} a0=0x{:x} a1=0x{:x}",
		ctx.host_functions.len(),
		frame.a0,
		frame.a1,
	);
	if index >= ctx.host_functions.len() {
		ctx.last_error = Some(format!("host function index out of range: {index}"));
		frame.a7 = u64::MAX;
		return;
	}

	// --- Handle grow_heap specially ---
	if ctx.grow_heap_index == Some(index) {
		let size = frame.a0 as u32;
		let new_heap_size = ctx.heap_top.wrapping_sub(ctx.heap_base).wrapping_add(size);
		if new_heap_size > ctx.max_heap_size {
			log::warn!(
				target: crate::LOG_TARGET,
				"grow_heap failed: requested size={size}, \
				 current_top=0x{:x}, base=0x{:x}, max=0x{:x}",
				ctx.heap_top, ctx.heap_base, ctx.max_heap_size
			);
			frame.a0 = 0; // failure
		} else {
			let new_top = ctx.heap_base.wrapping_add(new_heap_size);
			ctx.heap_top = new_top;
			frame.a0 = new_top as u64;
		}
		return;
	}

	// --- Look up host function ---
	let function = match ctx.host_functions[index] {
		Some(f) => f,
		None => {
			ctx.last_error =
				Some(format!("called unresolved import at index {index}"));
			frame.a7 = u64::MAX;
			return;
		},
	};
	log::trace!(target: crate::LOG_TARGET, "dispatching to '{}'", function.name());

	// Build argument values from registers a0-a5
	let mut args = [Value::I64(0); 6];
	let mut nth_reg = 0;
	for (nth_arg, kind) in function.signature().args.iter().enumerate() {
		match kind {
			ValueType::I32 => {
				let reg_val = get_arg_reg(frame, nth_reg);
				args[nth_arg] = Value::I32(reg_val as i32);
				nth_reg += 1;
			},
			ValueType::F32 => {
				let reg_val = get_arg_reg(frame, nth_reg);
				args[nth_arg] = Value::F32(reg_val as u32);
				nth_reg += 1;
			},
			ValueType::I64 => {
				let reg_val = get_arg_reg(frame, nth_reg);
				args[nth_arg] = Value::I64(reg_val as i64);
				nth_reg += 1;
			},
			ValueType::F64 => {
				let reg_val = get_arg_reg(frame, nth_reg);
				args[nth_arg] = Value::F64(reg_val);
				nth_reg += 1;
			},
		}
	}

	log::trace!(
		target: crate::LOG_TARGET,
		"calling host function: '{}', args = {:?}",
		function.name(),
		&args[..function.signature().args.len()]
	);

	let memory = &mut *ctx.memory;
	let mut func_ctx = NativeFunctionContext {
		memory,
		input_data: &mut ctx.input_data,
		frame,
	};

	let value = match function.execute(
		&mut func_ctx,
		&mut args.into_iter().take(function.signature().args.len()),
	) {
		Ok(value) => value,
		Err(error) => {
			ctx.last_error =
				Some(format!("call into host function '{}' failed: {error}", function.name()));
			return;
		},
	};

	// Write return value back to registers
	let frame = &mut *frame_ptr; // reborrow after func_ctx drop
	if let Some(value) = value {
		match value {
			Value::I32(v) => {
				frame.a0 = v as u64;
			},
			Value::F32(v) => {
				frame.a0 = v as u64;
			},
			Value::I64(v) => {
				frame.a0 = v as u64;
			},
			Value::F64(v) => {
				frame.a0 = v;
			},
		}
	}
}

/// Read an argument register from the saved frame.
fn get_arg_reg(frame: &SavedFrame, index: usize) -> u64 {
	match index {
		0 => frame.a0,
		1 => frame.a1,
		2 => frame.a2,
		3 => frame.a3,
		4 => frame.a4,
		5 => frame.a5,
		_ => panic!("too many host function arguments (max 6); qed"),
	}
}

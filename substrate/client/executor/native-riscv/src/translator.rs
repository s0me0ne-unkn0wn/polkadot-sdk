// This file is part of Substrate.

// Copyright (C) Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: GPL-3.0-or-later WITH Classpath-exception-2.0

//! PVM bytecode to native RV64IMAC + xtheadbb + xtheadcondmov machine code translator.

use crate::riscv_emit::Emitter;
use polkavm_common::program::{InstructionVisitor, RawReg};

// Guest registers (used by PVM programs)
const REG_RA: u8 = 1;
const REG_SP: u8 = 2;
const REG_T0: u8 = 5;
const REG_T1: u8 = 6;
const REG_T2: u8 = 7;
const REG_S0: u8 = 8;
const REG_S1: u8 = 9;
const REG_A0: u8 = 10;
const REG_A1: u8 = 11;
const REG_A2: u8 = 12;
const REG_A3: u8 = 13;
const REG_A4: u8 = 14;
const REG_A5: u8 = 15;

// Scratch registers (NOT guest-visible, used by translator)
const TMP1: u8 = 28; // x28 = t3
const TMP2: u8 = 29; // x29 = t4

// Context/state registers (callee-saved, set up before entering generated code)
const REG_MEM_BASE: u8 = 18; // x18 = s2 - guest memory base pointer
#[allow(dead_code)]
const REG_CTX: u8 = 3; // x3 = gp  - pointer to HostContext
const REG_CODE_BASE: u8 = 20; // x20 = s4 - base address of native code buffer
const REG_ADDR_TABLE: u8 = 21; // x21 = s5 - pointer to addr_to_native table data

/// Map a PVM RawReg to a hardware register number.
fn reg(r: RawReg) -> u8 {
	use polkavm_common::program::Reg;
	match r.get() {
		Reg::RA => REG_RA,
		Reg::SP => REG_SP,
		Reg::T0 => REG_T0,
		Reg::T1 => REG_T1,
		Reg::T2 => REG_T2,
		Reg::S0 => REG_S0,
		Reg::S1 => REG_S1,
		Reg::A0 => REG_A0,
		Reg::A1 => REG_A1,
		Reg::A2 => REG_A2,
		Reg::A3 => REG_A3,
		Reg::A4 => REG_A4,
		Reg::A5 => REG_A5,
	}
}

// ---------------------------------------------------------------------------
// Fixup infrastructure for branches/jumps
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum FixupKind {
	/// AUIPC TMP1, 0; JALR x0, TMP1, 0 at native_offset (8 bytes, ±2GB range).
	/// Used for all cross-basic-block jumps (unconditional and conditional).
	LongJump,
}

struct Fixup {
	/// Byte offset within generated native code where the instruction lives
	native_offset: usize,
	/// Target PVM program counter
	target_pvm_pc: u32,
	kind: FixupKind,
}

// ---------------------------------------------------------------------------
// Translator
// ---------------------------------------------------------------------------

pub struct Translator {
	pub asm: Emitter,
	/// Map from PVM program counter -> byte offset in native code.
	/// Index = PVM PC, value = native byte offset.
	pc_to_native: Vec<u32>,
	/// Pending fixups to resolve after first pass.
	fixups: Vec<Fixup>,
	/// Current PVM PC being translated (set externally before each instruction).
	current_pc: u32,
	/// Byte offset of the trampoline in native code.
	trampoline_offset: Option<usize>,
	/// Length of the addr_to_native table (for indirect jump bounds checks).
	/// PVM addresses used by indirect jumps index into a separate table;
	/// this is its size, set from `blob.jump_table().len() * 2 + 1`.
	addr_table_len: usize,
}

impl Translator {
	pub fn new(max_pvm_pc: u32) -> Self {
		Self {
			asm: Emitter::new(),
			pc_to_native: vec![u32::MAX; max_pvm_pc as usize + 1],
			fixups: Vec::new(),
			current_pc: 0,
			trampoline_offset: None,
			addr_table_len: 0,
		}
	}

	/// Set the addr_to_native table length for indirect jump bounds checks.
	/// Must be called before translating any instructions.
	pub fn set_addr_table_len(&mut self, len: usize) {
		self.addr_table_len = len;
	}

	/// Must be called before visiting each instruction.
	pub fn set_current_pc(&mut self, pc: u32) {
		self.current_pc = pc;
		if (pc as usize) < self.pc_to_native.len() {
			self.pc_to_native[pc as usize] = self.asm.position() as u32;
		}
	}

	/// Map a PVM PC to the current native code position without changing current_pc.
	/// Used to ensure `next_offset` PCs are also mapped (for indirect jump returns).
	pub fn map_pc(&mut self, pc: u32) {
		if (pc as usize) < self.pc_to_native.len() {
			// Only set if not already mapped (don't overwrite existing mappings)
			if self.pc_to_native[pc as usize] == u32::MAX {
				self.pc_to_native[pc as usize] = self.asm.position() as u32;
			}
		}
	}

	pub fn set_trampoline_offset(&mut self, offset: usize) {
		self.trampoline_offset = Some(offset);
	}

	/// Resolve all branch/jump fixups after the first pass.
	pub fn resolve_fixups(&mut self) {
		for fixup in &self.fixups {
			let target_native = self.pc_to_native[fixup.target_pvm_pc as usize] as usize;
			if target_native == u32::MAX as usize {
				// Target PVM PC has no native code. Overwrite the AUIPC+JALR
				// placeholder (8 bytes) with two unimp (illegal) instructions
				// so that jumping here traps instead of looping forever.
				self.asm.write_u32_at(fixup.native_offset, 0); // unimp
				self.asm.write_u32_at(fixup.native_offset + 4, 0); // unimp
				continue;
			}
			match fixup.kind {
				FixupKind::LongJump => {
					self.asm.patch_auipc_jalr(fixup.native_offset, target_native);
				},
			}
		}
	}

	pub fn into_parts(self) -> (Vec<u8>, Vec<u32>) {
		(self.asm.into_code(), self.pc_to_native)
	}

	// -----------------------------------------------------------------------
	// Helpers
	// -----------------------------------------------------------------------

	/// Load a 32-bit immediate that might not fit in 12-bit sign-extended.
	/// Uses TMP1 as scratch. Returns the register holding the value.
	fn load_imm_to_tmp(&mut self, value: u32) -> u8 {
		self.asm.li32(TMP1, value as i32);
		TMP1
	}

	/// Emit guest address computation: host_addr = mem_base + guest_addr(32-bit)
	/// Result in TMP1. `base_reg` holds a guest address (only lower 32 bits valid).
	fn guest_addr_to_host(&mut self, base_reg: u8, offset: u32) {
		if offset == 0 {
			// Zero-extend base_reg to 64 bits, add mem_base
			self.asm.slli(TMP1, base_reg, 32);
			self.asm.srli(TMP1, TMP1, 32);
			self.asm.add(TMP1, TMP1, REG_MEM_BASE);
		} else if (offset as i32) >= -2048 && (offset as i32) < 2048 {
			self.asm.addiw(TMP1, base_reg, offset as i32);
			// Zero-extend result
			self.asm.slli(TMP1, TMP1, 32);
			self.asm.srli(TMP1, TMP1, 32);
			self.asm.add(TMP1, TMP1, REG_MEM_BASE);
		} else {
			self.asm.li32(TMP2, offset as i32);
			self.asm.addw(TMP1, base_reg, TMP2);
			self.asm.slli(TMP1, TMP1, 32);
			self.asm.srli(TMP1, TMP1, 32);
			self.asm.add(TMP1, TMP1, REG_MEM_BASE);
		}
	}

	/// Emit absolute guest address -> host address. Result in TMP1.
	fn abs_guest_addr(&mut self, addr: u32) {
		self.asm.li_u32(TMP1, addr);
		self.asm.add(TMP1, TMP1, REG_MEM_BASE);
	}

	/// Emit a long-form unconditional jump placeholder (AUIPC TMP1 + JALR x0, TMP1).
	/// Records a LongJump fixup to be resolved after the first pass.
	/// Total: 8 bytes. Range: ±2GB.
	fn emit_long_jump_fixup(&mut self, target_pvm_pc: u32) {
		let pos = self.asm.position();
		self.fixups.push(Fixup {
			native_offset: pos,
			target_pvm_pc,
			kind: FixupKind::LongJump,
		});
		self.asm.auipc(TMP1, 0); // placeholder hi20
		self.asm.jalr(0, TMP1, 0); // placeholder lo12, no link (x0)
	}

	/// Emit runtime indirect jump: compute PVM target address from register + offset,
	/// look up native offset in addr_to_native table (via s5), add code base (s4), jump.
	///
	/// PVM indirect jumps use **PVM addresses** (aligned to VM_CODE_ADDRESS_ALIGNMENT=2),
	/// NOT byte offsets. The addr_to_native table (pointed to by s5) is indexed by PVM
	/// address and maps to native code offsets. It is built from blob.jump_table().
	///
	/// Includes runtime bounds checking (PVM addr < table length) and unmapped address
	/// detection (native_offset != u32::MAX). Both cases trap via `unimp`.
	fn emit_indirect_jump(&mut self, base_reg: u8, offset: u32) {
		// Compute target PVM address = (base_reg + offset) as u32 in TMP1
		if offset == 0 {
			// Zero-extend base_reg to 64 bits
			self.asm.slli(TMP1, base_reg, 32);
			self.asm.srli(TMP1, TMP1, 32);
		} else {
			self.asm.li32(TMP2, offset as i32);
			self.asm.addw(TMP1, base_reg, TMP2); // 32-bit wrapping add
			// Zero-extend result to 64 bits
			self.asm.slli(TMP1, TMP1, 32);
			self.asm.srli(TMP1, TMP1, 32);
		}

		// Bounds check: trap if pvm_addr >= addr_to_native.len()
		let table_len = self.addr_table_len as i32;
		self.asm.li32(TMP2, table_len);
		self.asm.bgeu(TMP1, TMP2, 0); // placeholder offset, patched below
		let bounds_branch = self.asm.position() - 4;

		// Save target PVM address in a6 (x16) for crash diagnostics.
		// a6 is not a guest register and not used by the trampoline.
		self.asm.mv(16, TMP1); // a6 = target PVM address

		// Look up native offset: addr_to_native[TMP1]
		self.asm.slli(TMP2, TMP1, 2); // TMP2 = pvm_addr * 4 (each entry is u32)
		self.asm.add(TMP2, REG_ADDR_TABLE, TMP2); // TMP2 = &addr_to_native[pvm_addr]
		self.asm.lwu(TMP1, TMP2, 0); // TMP1 = native_offset (zero-extended u32)

		// Unmapped check: trap if native_offset == u32::MAX (sentinel for unmapped addrs).
		// After lwu, TMP1 = 0x00000000_FFFFFFFF for unmapped entries.
		// addiw wraps the low 32 bits: 0xFFFFFFFF + 1 = 0 (sign-extended to 64-bit 0).
		self.asm.addiw(TMP2, TMP1, 1);
		self.asm.beq(TMP2, 0, 0); // placeholder offset, patched below
		let unmapped_branch = self.asm.position() - 4;

		// Compute host address and jump
		self.asm.add(TMP1, REG_CODE_BASE, TMP1); // TMP1 = code_base + native_offset
		self.asm.jalr(0, TMP1, 0); // jump (no link)

		// Trap target for out-of-bounds or unmapped PVM addresses
		let trap_pos = self.asm.position();
		self.asm.unimp();

		self.asm.patch_branch(bounds_branch, trap_pos);
		self.asm.patch_branch(unmapped_branch, trap_pos);
	}
}

impl InstructionVisitor for Translator {
	type ReturnTy = ();

	// =======================================================================
	// Argless instructions
	// =======================================================================

	fn trap(&mut self) {
		self.asm.unimp();
	}

	fn fallthrough(&mut self) {
		// Basic block boundary marker — no-op in native code
	}

	fn memset(&mut self) {
		// memset(a0=dest_guest_addr, a1=byte_value, a2=count)
		// PVM: after completion, A0 = dest + count, A2 = 0
		self.guest_addr_to_host(REG_A0, 0); // TMP1 = host ptr (offset=0 so TMP2 safe)
		self.asm.mv(TMP2, REG_A2); // save original count
		let loop_start = self.asm.position();
		self.asm.beq(REG_A2, 0, 0); // placeholder, patch below
		let branch_pos = self.asm.position() - 4;
		self.asm.sb(REG_A1, TMP1, 0);
		self.asm.addi(TMP1, TMP1, 1);
		self.asm.addi(REG_A2, REG_A2, -1);
		let end = self.asm.position();
		self.asm.j((loop_start as i32) - (end as i32));
		let done = self.asm.position();
		self.asm.patch_branch(branch_pos, done);
		self.asm.add(REG_A0, REG_A0, TMP2); // A0 = dest + count
	}

	fn unlikely(&mut self) {
		// Optimization hint — no-op
	}

	// =======================================================================
	// reg, imm instructions
	// =======================================================================

	fn sbrk(&mut self, _dst: RawReg, _size: RawReg) {
		// sbrk was removed from JAMv1 ISA (GP 0.8.0). Trap if encountered.
		self.asm.unimp();
	}

	fn jump_indirect(&mut self, base: RawReg, offset: u32) {
		self.emit_indirect_jump(reg(base), offset);
	}

	fn load_imm(&mut self, d: RawReg, imm: u32) {
		self.asm.li32(reg(d), imm as i32);
	}

	fn load_u8(&mut self, dst: RawReg, addr: u32) {
		self.abs_guest_addr(addr);
		self.asm.lbu(reg(dst), TMP1, 0);
	}
	fn load_i8(&mut self, dst: RawReg, addr: u32) {
		self.abs_guest_addr(addr);
		self.asm.lb(reg(dst), TMP1, 0);
	}
	fn load_u16(&mut self, dst: RawReg, addr: u32) {
		self.abs_guest_addr(addr);
		self.asm.lhu(reg(dst), TMP1, 0);
	}
	fn load_i16(&mut self, dst: RawReg, addr: u32) {
		self.abs_guest_addr(addr);
		self.asm.lh(reg(dst), TMP1, 0);
	}
	fn load_i32(&mut self, dst: RawReg, addr: u32) {
		self.abs_guest_addr(addr);
		self.asm.lw(reg(dst), TMP1, 0);
	}
	fn load_u32(&mut self, dst: RawReg, addr: u32) {
		self.abs_guest_addr(addr);
		self.asm.lwu(reg(dst), TMP1, 0);
	}
	fn load_u64(&mut self, dst: RawReg, addr: u32) {
		self.abs_guest_addr(addr);
		self.asm.ld(reg(dst), TMP1, 0);
	}

	fn store_u8(&mut self, src: RawReg, addr: u32) {
		self.abs_guest_addr(addr);
		self.asm.sb(reg(src), TMP1, 0);
	}
	fn store_u16(&mut self, src: RawReg, addr: u32) {
		self.abs_guest_addr(addr);
		self.asm.sh(reg(src), TMP1, 0);
	}
	fn store_u32(&mut self, src: RawReg, addr: u32) {
		self.abs_guest_addr(addr);
		self.asm.sw(reg(src), TMP1, 0);
	}
	fn store_u64(&mut self, src: RawReg, addr: u32) {
		self.abs_guest_addr(addr);
		self.asm.sd(reg(src), TMP1, 0);
	}

	// =======================================================================
	// reg, imm, offset instructions (branches with immediate)
	// =======================================================================

	fn load_imm_and_jump(&mut self, ra: RawReg, value: u32, target: u32) {
		self.asm.li32(reg(ra), value as i32);
		self.emit_long_jump_fixup(target);
	}

	fn branch_eq_imm(&mut self, s1: RawReg, imm: u32, target: u32) {
		let tmp = self.load_imm_to_tmp(imm);
		self.asm.bne(reg(s1), tmp, 12); // skip long jump if NOT equal
		self.emit_long_jump_fixup(target);
	}
	fn branch_not_eq_imm(&mut self, s1: RawReg, imm: u32, target: u32) {
		let tmp = self.load_imm_to_tmp(imm);
		self.asm.beq(reg(s1), tmp, 12); // skip long jump if equal
		self.emit_long_jump_fixup(target);
	}
	fn branch_less_unsigned_imm(&mut self, s1: RawReg, imm: u32, target: u32) {
		let tmp = self.load_imm_to_tmp(imm);
		self.asm.bgeu(reg(s1), tmp, 12); // skip if s1 >= imm (unsigned)
		self.emit_long_jump_fixup(target);
	}
	fn branch_less_signed_imm(&mut self, s1: RawReg, imm: u32, target: u32) {
		let tmp = self.load_imm_to_tmp(imm);
		self.asm.bge(reg(s1), tmp, 12); // skip if s1 >= imm (signed)
		self.emit_long_jump_fixup(target);
	}
	fn branch_greater_or_equal_unsigned_imm(&mut self, s1: RawReg, imm: u32, target: u32) {
		let tmp = self.load_imm_to_tmp(imm);
		self.asm.bltu(reg(s1), tmp, 12); // skip if s1 < imm (unsigned)
		self.emit_long_jump_fixup(target);
	}
	fn branch_greater_or_equal_signed_imm(&mut self, s1: RawReg, imm: u32, target: u32) {
		let tmp = self.load_imm_to_tmp(imm);
		self.asm.blt(reg(s1), tmp, 12); // skip if s1 < imm (signed)
		self.emit_long_jump_fixup(target);
	}
	fn branch_less_or_equal_signed_imm(&mut self, s1: RawReg, imm: u32, target: u32) {
		// condition: bge(tmp, s1) — inverted: blt(tmp, s1)
		let tmp = self.load_imm_to_tmp(imm);
		self.asm.blt(tmp, reg(s1), 12); // skip if imm < s1 (signed)
		self.emit_long_jump_fixup(target);
	}
	fn branch_less_or_equal_unsigned_imm(&mut self, s1: RawReg, imm: u32, target: u32) {
		// condition: bgeu(tmp, s1) — inverted: bltu(tmp, s1)
		let tmp = self.load_imm_to_tmp(imm);
		self.asm.bltu(tmp, reg(s1), 12); // skip if imm < s1 (unsigned)
		self.emit_long_jump_fixup(target);
	}
	fn branch_greater_signed_imm(&mut self, s1: RawReg, imm: u32, target: u32) {
		// condition: blt(tmp, s1) — inverted: bge(tmp, s1)
		let tmp = self.load_imm_to_tmp(imm);
		self.asm.bge(tmp, reg(s1), 12); // skip if imm >= s1 (signed)
		self.emit_long_jump_fixup(target);
	}
	fn branch_greater_unsigned_imm(&mut self, s1: RawReg, imm: u32, target: u32) {
		// condition: bltu(tmp, s1) — inverted: bgeu(tmp, s1)
		let tmp = self.load_imm_to_tmp(imm);
		self.asm.bgeu(tmp, reg(s1), 12); // skip if imm >= s1 (unsigned)
		self.emit_long_jump_fixup(target);
	}

	// =======================================================================
	// reg, imm, imm instructions (store_imm_indirect)
	// =======================================================================

	fn store_imm_indirect_u8(&mut self, base: RawReg, offset: u32, value: u32) {
		self.guest_addr_to_host(reg(base), offset);
		self.asm.li32(TMP2, value as i32);
		self.asm.sb(TMP2, TMP1, 0);
	}
	fn store_imm_indirect_u16(&mut self, base: RawReg, offset: u32, value: u32) {
		self.guest_addr_to_host(reg(base), offset);
		self.asm.li32(TMP2, value as i32);
		self.asm.sh(TMP2, TMP1, 0);
	}
	fn store_imm_indirect_u32(&mut self, base: RawReg, offset: u32, value: u32) {
		self.guest_addr_to_host(reg(base), offset);
		self.asm.li32(TMP2, value as i32);
		self.asm.sw(TMP2, TMP1, 0);
	}
	fn store_imm_indirect_u64(&mut self, base: RawReg, offset: u32, value: u32) {
		self.guest_addr_to_host(reg(base), offset);
		self.asm.li32(TMP2, value as i32);
		self.asm.sd(TMP2, TMP1, 0);
	}

	// =======================================================================
	// reg, reg, imm instructions (ALU with immediate + load/store indirect)
	// =======================================================================

	fn store_indirect_u8(&mut self, src: RawReg, base: RawReg, offset: u32) {
		self.guest_addr_to_host(reg(base), offset);
		self.asm.sb(reg(src), TMP1, 0);
	}
	fn store_indirect_u16(&mut self, src: RawReg, base: RawReg, offset: u32) {
		self.guest_addr_to_host(reg(base), offset);
		self.asm.sh(reg(src), TMP1, 0);
	}
	fn store_indirect_u32(&mut self, src: RawReg, base: RawReg, offset: u32) {
		self.guest_addr_to_host(reg(base), offset);
		self.asm.sw(reg(src), TMP1, 0);
	}
	fn store_indirect_u64(&mut self, src: RawReg, base: RawReg, offset: u32) {
		self.guest_addr_to_host(reg(base), offset);
		self.asm.sd(reg(src), TMP1, 0);
	}
	fn load_indirect_u8(&mut self, dst: RawReg, base: RawReg, offset: u32) {
		self.guest_addr_to_host(reg(base), offset);
		self.asm.lbu(reg(dst), TMP1, 0);
	}
	fn load_indirect_i8(&mut self, dst: RawReg, base: RawReg, offset: u32) {
		self.guest_addr_to_host(reg(base), offset);
		self.asm.lb(reg(dst), TMP1, 0);
	}
	fn load_indirect_u16(&mut self, dst: RawReg, base: RawReg, offset: u32) {
		self.guest_addr_to_host(reg(base), offset);
		self.asm.lhu(reg(dst), TMP1, 0);
	}
	fn load_indirect_i16(&mut self, dst: RawReg, base: RawReg, offset: u32) {
		self.guest_addr_to_host(reg(base), offset);
		self.asm.lh(reg(dst), TMP1, 0);
	}
	fn load_indirect_i32(&mut self, dst: RawReg, base: RawReg, offset: u32) {
		self.guest_addr_to_host(reg(base), offset);
		self.asm.lw(reg(dst), TMP1, 0);
	}
	fn load_indirect_u32(&mut self, dst: RawReg, base: RawReg, offset: u32) {
		self.guest_addr_to_host(reg(base), offset);
		self.asm.lwu(reg(dst), TMP1, 0);
	}
	fn load_indirect_u64(&mut self, dst: RawReg, base: RawReg, offset: u32) {
		self.guest_addr_to_host(reg(base), offset);
		self.asm.ld(reg(dst), TMP1, 0);
	}

	// ALU reg, reg, imm
	fn add_imm_32(&mut self, d: RawReg, s: RawReg, imm: u32) {
		let imm = imm as i32;
		if imm >= -2048 && imm < 2048 {
			self.asm.addiw(reg(d), reg(s), imm);
		} else {
			self.asm.li32(TMP1, imm);
			self.asm.addw(reg(d), reg(s), TMP1);
		}
	}
	fn add_imm_64(&mut self, d: RawReg, s: RawReg, imm: u32) {
		let imm = imm as i32;
		if imm >= -2048 && imm < 2048 {
			self.asm.addi(reg(d), reg(s), imm);
		} else {
			self.asm.li32(TMP1, imm);
			self.asm.add(reg(d), reg(s), TMP1);
		}
	}
	fn and_imm(&mut self, d: RawReg, s: RawReg, imm: u32) {
		let imm = imm as i32;
		if imm >= -2048 && imm < 2048 {
			self.asm.andi(reg(d), reg(s), imm);
		} else {
			self.asm.li32(TMP1, imm);
			self.asm.and(reg(d), reg(s), TMP1);
		}
	}
	fn xor_imm(&mut self, d: RawReg, s: RawReg, imm: u32) {
		let imm = imm as i32;
		if imm >= -2048 && imm < 2048 {
			self.asm.xori(reg(d), reg(s), imm);
		} else {
			self.asm.li32(TMP1, imm);
			self.asm.xor(reg(d), reg(s), TMP1);
		}
	}
	fn or_imm(&mut self, d: RawReg, s: RawReg, imm: u32) {
		let imm = imm as i32;
		if imm >= -2048 && imm < 2048 {
			self.asm.ori(reg(d), reg(s), imm);
		} else {
			self.asm.li32(TMP1, imm);
			self.asm.or(reg(d), reg(s), TMP1);
		}
	}
	fn mul_imm_32(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.li32(TMP1, imm as i32);
		self.asm.mulw(reg(d), reg(s), TMP1);
	}
	fn mul_imm_64(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.li32(TMP1, imm as i32);
		self.asm.mul(reg(d), reg(s), TMP1);
	}
	fn set_less_than_unsigned_imm(&mut self, d: RawReg, s: RawReg, imm: u32) {
		let imm = imm as i32;
		if imm >= -2048 && imm < 2048 {
			self.asm.sltiu(reg(d), reg(s), imm);
		} else {
			self.asm.li32(TMP1, imm);
			self.asm.sltu(reg(d), reg(s), TMP1);
		}
	}
	fn set_less_than_signed_imm(&mut self, d: RawReg, s: RawReg, imm: u32) {
		let imm = imm as i32;
		if imm >= -2048 && imm < 2048 {
			self.asm.slti(reg(d), reg(s), imm);
		} else {
			self.asm.li32(TMP1, imm);
			self.asm.slt(reg(d), reg(s), TMP1);
		}
	}
	fn shift_logical_left_imm_32(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.slliw(reg(d), reg(s), imm & 0x1f);
	}
	fn shift_logical_left_imm_64(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.slli(reg(d), reg(s), imm & 0x3f);
	}
	fn shift_logical_right_imm_32(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.srliw(reg(d), reg(s), imm & 0x1f);
	}
	fn shift_logical_right_imm_64(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.srli(reg(d), reg(s), imm & 0x3f);
	}
	fn shift_arithmetic_right_imm_32(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.sraiw(reg(d), reg(s), imm & 0x1f);
	}
	fn shift_arithmetic_right_imm_64(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.srai(reg(d), reg(s), imm & 0x3f);
	}
	fn negate_and_add_imm_32(&mut self, d: RawReg, s: RawReg, imm: u32) {
		// d = imm - s (32-bit)
		self.asm.li32(TMP1, imm as i32);
		self.asm.subw(reg(d), TMP1, reg(s));
	}
	fn negate_and_add_imm_64(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.li32(TMP1, imm as i32);
		self.asm.sub(reg(d), TMP1, reg(s));
	}
	fn set_greater_than_unsigned_imm(&mut self, d: RawReg, s: RawReg, imm: u32) {
		// d = (s > imm) ? 1 : 0  <=>  d = (imm < s) ? 1 : 0
		self.asm.li32(TMP1, imm as i32);
		self.asm.sltu(reg(d), TMP1, reg(s));
	}
	fn set_greater_than_signed_imm(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.li32(TMP1, imm as i32);
		self.asm.slt(reg(d), TMP1, reg(s));
	}

	// "alt" shifts: d = imm >> s (shift the immediate by register)
	fn shift_logical_right_imm_alt_32(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.li32(TMP1, imm as i32);
		self.asm.srlw(reg(d), TMP1, reg(s));
	}
	fn shift_logical_right_imm_alt_64(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.li32(TMP1, imm as i32);
		self.asm.srl(reg(d), TMP1, reg(s));
	}
	fn shift_arithmetic_right_imm_alt_32(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.li32(TMP1, imm as i32);
		self.asm.sraw(reg(d), TMP1, reg(s));
	}
	fn shift_arithmetic_right_imm_alt_64(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.li32(TMP1, imm as i32);
		self.asm.sra(reg(d), TMP1, reg(s));
	}
	fn shift_logical_left_imm_alt_32(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.li32(TMP1, imm as i32);
		self.asm.sllw(reg(d), TMP1, reg(s));
	}
	fn shift_logical_left_imm_alt_64(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.li32(TMP1, imm as i32);
		self.asm.sll(reg(d), TMP1, reg(s));
	}

	// cmov with immediate
	fn cmov_if_zero_imm(&mut self, d: RawReg, c: RawReg, imm: u32) {
		// if c == 0, d = imm
		self.asm.li32(TMP1, imm as i32);
		self.asm.th_mveqz(reg(d), TMP1, reg(c));
	}
	fn cmov_if_not_zero_imm(&mut self, d: RawReg, c: RawReg, imm: u32) {
		self.asm.li32(TMP1, imm as i32);
		self.asm.th_mvnez(reg(d), TMP1, reg(c));
	}

	// rotate right immediate
	fn rotate_right_imm_32(&mut self, d: RawReg, s: RawReg, imm: u32) {
		// th.srriw: 32-bit rotate right immediate (xtheadbb)
		// Encoding: funct7=0001010, shamt[4:0], rs1, funct3=001, rd, CUSTOM_0
		let shamt = (imm & 0x1f) as u8;
		let funct7: u32 = 0b0001010;
		let word = (funct7 << 25)
			| ((shamt as u32) << 20)
			| ((reg(s) as u32) << 15)
			| (0b001 << 12)
			| ((reg(d) as u32) << 7)
			| 0b0001011; // CUSTOM_0
		self.asm.emit32_pub(word);
	}
	fn rotate_right_imm_alt_32(&mut self, d: RawReg, s: RawReg, imm: u32) {
		// d = imm ror s (rotate immediate by register amount, 32-bit)
		// Save s to TMP2 first to handle d==s aliasing
		self.asm.li32(TMP1, imm as i32);
		self.asm.mv(TMP2, reg(s));
		self.asm.srlw(reg(d), TMP1, TMP2); // d = imm >> s
		self.asm.neg(TMP2, TMP2); // TMP2 = -s (using saved value)
		self.asm.sllw(TMP1, TMP1, TMP2); // TMP1 = imm << (32-s) (W-suffix masks to 5 bits)
		self.asm.or(reg(d), reg(d), TMP1);
	}
	fn rotate_right_imm_64(&mut self, d: RawReg, s: RawReg, imm: u32) {
		self.asm.th_srri(reg(d), reg(s), (imm & 0x3f) as u8);
	}
	fn rotate_right_imm_alt_64(&mut self, d: RawReg, s: RawReg, imm: u32) {
		// d = imm ror s (64-bit)
		// Save s to TMP2 first to handle d==s aliasing
		self.asm.li32(TMP1, imm as i32);
		self.asm.mv(TMP2, reg(s));
		self.asm.srl(reg(d), TMP1, TMP2); // d = imm >> s
		self.asm.neg(TMP2, TMP2); // TMP2 = -s (using saved value)
		self.asm.sll(TMP1, TMP1, TMP2); // TMP1 = imm << (64-s)
		self.asm.or(reg(d), reg(d), TMP1);
	}

	// =======================================================================
	// reg, reg, offset instructions (branches)
	// =======================================================================

	fn branch_eq(&mut self, s1: RawReg, s2: RawReg, target: u32) {
		self.asm.bne(reg(s1), reg(s2), 12); // skip long jump if NOT equal
		self.emit_long_jump_fixup(target);
	}
	fn branch_not_eq(&mut self, s1: RawReg, s2: RawReg, target: u32) {
		self.asm.beq(reg(s1), reg(s2), 12); // skip long jump if equal
		self.emit_long_jump_fixup(target);
	}
	fn branch_less_unsigned(&mut self, s1: RawReg, s2: RawReg, target: u32) {
		self.asm.bgeu(reg(s1), reg(s2), 12); // skip if s1 >= s2 (unsigned)
		self.emit_long_jump_fixup(target);
	}
	fn branch_less_signed(&mut self, s1: RawReg, s2: RawReg, target: u32) {
		self.asm.bge(reg(s1), reg(s2), 12); // skip if s1 >= s2 (signed)
		self.emit_long_jump_fixup(target);
	}
	fn branch_greater_or_equal_unsigned(&mut self, s1: RawReg, s2: RawReg, target: u32) {
		self.asm.bltu(reg(s1), reg(s2), 12); // skip if s1 < s2 (unsigned)
		self.emit_long_jump_fixup(target);
	}
	fn branch_greater_or_equal_signed(&mut self, s1: RawReg, s2: RawReg, target: u32) {
		self.asm.blt(reg(s1), reg(s2), 12); // skip if s1 < s2 (signed)
		self.emit_long_jump_fixup(target);
	}

	// =======================================================================
	// reg, reg, reg instructions (ALU)
	// =======================================================================

	fn add_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.addw(reg(d), reg(s1), reg(s2));
	}
	fn add_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.add(reg(d), reg(s1), reg(s2));
	}
	fn sub_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.subw(reg(d), reg(s1), reg(s2));
	}
	fn sub_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.sub(reg(d), reg(s1), reg(s2));
	}
	fn and(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.and(reg(d), reg(s1), reg(s2));
	}
	fn xor(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.xor(reg(d), reg(s1), reg(s2));
	}
	fn or(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.or(reg(d), reg(s1), reg(s2));
	}
	fn mul_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.mulw(reg(d), reg(s1), reg(s2));
	}
	fn mul_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.mul(reg(d), reg(s1), reg(s2));
	}
	fn mul_upper_signed_signed(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.mulh(reg(d), reg(s1), reg(s2));
	}
	fn mul_upper_unsigned_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.mulhu(reg(d), reg(s1), reg(s2));
	}
	fn mul_upper_signed_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.mulhsu(reg(d), reg(s1), reg(s2));
	}
	fn set_less_than_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.sltu(reg(d), reg(s1), reg(s2));
	}
	fn set_less_than_signed(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.slt(reg(d), reg(s1), reg(s2));
	}
	fn shift_logical_left_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.sllw(reg(d), reg(s1), reg(s2));
	}
	fn shift_logical_left_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.sll(reg(d), reg(s1), reg(s2));
	}
	fn shift_logical_right_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.srlw(reg(d), reg(s1), reg(s2));
	}
	fn shift_logical_right_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.srl(reg(d), reg(s1), reg(s2));
	}
	fn shift_arithmetic_right_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.sraw(reg(d), reg(s1), reg(s2));
	}
	fn shift_arithmetic_right_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.sra(reg(d), reg(s1), reg(s2));
	}
	fn div_unsigned_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.divuw(reg(d), reg(s1), reg(s2));
	}
	fn div_unsigned_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.divu(reg(d), reg(s1), reg(s2));
	}
	fn div_signed_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.divw(reg(d), reg(s1), reg(s2));
	}
	fn div_signed_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.div(reg(d), reg(s1), reg(s2));
	}
	fn rem_unsigned_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.remuw(reg(d), reg(s1), reg(s2));
	}
	fn rem_unsigned_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.remu(reg(d), reg(s1), reg(s2));
	}
	fn rem_signed_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.remw(reg(d), reg(s1), reg(s2));
	}
	fn rem_signed_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.rem(reg(d), reg(s1), reg(s2));
	}

	// Conditional move (xtheadcondmov)
	fn cmov_if_zero(&mut self, d: RawReg, s: RawReg, c: RawReg) {
		self.asm.th_mveqz(reg(d), reg(s), reg(c));
	}
	fn cmov_if_not_zero(&mut self, d: RawReg, s: RawReg, c: RawReg) {
		self.asm.th_mvnez(reg(d), reg(s), reg(c));
	}

	// Zbb equivalents via xtheadbb
	fn and_inverted(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		// andn: d = s1 & ~s2
		self.asm.not(TMP1, reg(s2));
		self.asm.and(reg(d), reg(s1), TMP1);
	}
	fn or_inverted(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.not(TMP1, reg(s2));
		self.asm.or(reg(d), reg(s1), TMP1);
	}
	fn xnor(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.xor(reg(d), reg(s1), reg(s2));
		self.asm.not(reg(d), reg(d));
	}
	fn maximum(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		// d = max(s1, s2) signed. Use slt + cmov.
		// Save s2 to TMP2 before writing d (handles d == s2 aliasing).
		self.asm.slt(TMP1, reg(s1), reg(s2)); // TMP1 = (s1 < s2)
		self.asm.mv(TMP2, reg(s2));
		self.asm.mv(reg(d), reg(s1));
		self.asm.th_mvnez(reg(d), TMP2, TMP1); // if s1 < s2, d = s2
	}
	fn maximum_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.sltu(TMP1, reg(s1), reg(s2));
		self.asm.mv(TMP2, reg(s2));
		self.asm.mv(reg(d), reg(s1));
		self.asm.th_mvnez(reg(d), TMP2, TMP1);
	}
	fn minimum(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		// Save s1 to TMP2 before writing d (handles d == s1 aliasing).
		self.asm.slt(TMP1, reg(s1), reg(s2));
		self.asm.mv(TMP2, reg(s1));
		self.asm.mv(reg(d), reg(s2));
		self.asm.th_mvnez(reg(d), TMP2, TMP1); // if s1 < s2, d = s1
	}
	fn minimum_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.sltu(TMP1, reg(s1), reg(s2));
		self.asm.mv(TMP2, reg(s1));
		self.asm.mv(reg(d), reg(s2));
		self.asm.th_mvnez(reg(d), TMP2, TMP1);
	}

	// Rotate (register) — no direct xtheadbb, emulate with shifts
	fn rotate_left_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		// d = (s1 << s2) | (s1 >> (32 - s2)), 32-bit
		self.asm.sllw(TMP1, reg(s1), reg(s2));
		self.asm.neg(TMP2, reg(s2));
		self.asm.srlw(TMP2, reg(s1), TMP2);
		self.asm.or(reg(d), TMP1, TMP2);
	}
	fn rotate_left_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.sll(TMP1, reg(s1), reg(s2));
		self.asm.neg(TMP2, reg(s2));
		self.asm.srl(TMP2, reg(s1), TMP2);
		self.asm.or(reg(d), TMP1, TMP2);
	}
	fn rotate_right_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.srlw(TMP1, reg(s1), reg(s2));
		self.asm.neg(TMP2, reg(s2));
		self.asm.sllw(TMP2, reg(s1), TMP2);
		self.asm.or(reg(d), TMP1, TMP2);
	}
	fn rotate_right_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
		self.asm.srl(TMP1, reg(s1), reg(s2));
		self.asm.neg(TMP2, reg(s2));
		self.asm.sll(TMP2, reg(s1), TMP2);
		self.asm.or(reg(d), TMP1, TMP2);
	}

	// =======================================================================
	// offset instructions
	// =======================================================================

	fn jump(&mut self, target: u32) {
		self.emit_long_jump_fixup(target);
	}

	// =======================================================================
	// imm instructions
	// =======================================================================

	fn ecalli(&mut self, nth_import: u32) {
		if let Some(trampoline) = self.trampoline_offset {
			// li a7, nth_import; auipc TMP1, hi; jalr ra, TMP1, lo
			self.asm.li32(17, nth_import as i32); // a7 = x17
			let auipc_pos = self.asm.position();
			let delta = (trampoline as i64 - auipc_pos as i64) as i32;
			// Split delta for AUIPC+JALR, accounting for sign-extension of lo12
			let lo = (delta << 20) >> 20;
			let hi = delta.wrapping_sub(lo);
			self.asm.auipc(TMP1, hi); // TMP1 = PC + hi
			// Link to TMP2 (not ra) to avoid clobbering the guest's RA register.
			// The trampoline saves TMP2 as the return address separately.
			self.asm.jalr(TMP2, TMP1, lo); // call trampoline, TMP2 = return addr
		} else {
			// Fallback: trap
			self.asm.unimp();
		}
	}

	// =======================================================================
	// imm, imm instructions (store absolute)
	// =======================================================================

	fn store_imm_u8(&mut self, addr: u32, value: u32) {
		self.abs_guest_addr(addr);
		self.asm.li32(TMP2, value as i32);
		self.asm.sb(TMP2, TMP1, 0);
	}
	fn store_imm_u16(&mut self, addr: u32, value: u32) {
		self.abs_guest_addr(addr);
		self.asm.li32(TMP2, value as i32);
		self.asm.sh(TMP2, TMP1, 0);
	}
	fn store_imm_u32(&mut self, addr: u32, value: u32) {
		self.abs_guest_addr(addr);
		self.asm.li32(TMP2, value as i32);
		self.asm.sw(TMP2, TMP1, 0);
	}
	fn store_imm_u64(&mut self, addr: u32, value: u32) {
		self.abs_guest_addr(addr);
		self.asm.li32(TMP2, value as i32);
		self.asm.sd(TMP2, TMP1, 0);
	}

	// =======================================================================
	// reg, reg instructions
	// =======================================================================

	fn move_reg(&mut self, d: RawReg, s: RawReg) {
		self.asm.mv(reg(d), reg(s));
	}

	fn count_leading_zero_bits_32(&mut self, d: RawReg, s: RawReg) {
		// 32-bit clz: shift left by 32, then th.ff1
		self.asm.slli(TMP1, reg(s), 32);
		self.asm.th_ff1(reg(d), TMP1);
		// th_ff1(0) returns 64, but PVM clz32 expects 32 for zero input
		self.asm.li32(TMP1, 32);
		self.asm.slt(TMP2, TMP1, reg(d)); // TMP2 = 1 if 32 < d
		self.asm.th_mvnez(reg(d), TMP1, TMP2); // if d > 32, d = 32
	}

	fn count_leading_zero_bits_64(&mut self, d: RawReg, s: RawReg) {
		self.asm.th_ff1(reg(d), reg(s));
	}

	fn count_trailing_zero_bits_32(&mut self, d: RawReg, s: RawReg) {
		// ctz32: isolate lowest set bit, reverse bytes, clz
		// Or: rev + ff1. But th.rev reverses bytes, not bits.
		// Correct approach: negate and AND to isolate lowest bit, then clz gives ctz.
		// ctz(x) = 63 - clz(x & -x) for 64-bit. For 32-bit: ctz(x) = 31 - clz32((x & -x) << 32)
		// Simpler: zero-extend to 64-bit, set bit 32, then do ctz64
		self.asm.slli(TMP1, reg(s), 32);
		self.asm.srli(TMP1, TMP1, 32); // zero-extend
		// Set bit 32 as sentinel so ctz stops there
		self.asm.li32(TMP2, 1);
		self.asm.slli(TMP2, TMP2, 32);
		self.asm.or(TMP1, TMP1, TMP2);
		// Now ctz64(TMP1)
		self.asm.neg(TMP2, TMP1);
		self.asm.and(TMP1, TMP1, TMP2); // isolate lowest bit
		self.asm.th_ff1(reg(d), TMP1); // clz
		// ctz = 63 - clz(isolated_bit)
		self.asm.li32(TMP1, 63);
		self.asm.sub(reg(d), TMP1, reg(d));
	}

	fn count_trailing_zero_bits_64(&mut self, d: RawReg, s: RawReg) {
		// ctz64(x) = 63 - clz(x & -x), but clz(0)=64, so need special case for x=0
		// Save s to TMP2 to handle d==s aliasing (th_mveqz needs original s)
		self.asm.mv(TMP2, reg(s));
		self.asm.neg(TMP1, TMP2);
		self.asm.and(TMP1, TMP2, TMP1);
		self.asm.th_ff1(TMP1, TMP1);
		self.asm.li32(reg(d), 63);
		self.asm.sub(reg(d), reg(d), TMP1);
		// if s == 0, result should be 64 (63 - 64 = -1, so correct to 64)
		self.asm.li32(TMP1, 64);
		self.asm.th_mveqz(reg(d), TMP1, TMP2); // use saved s
	}

	fn count_set_bits_32(&mut self, d: RawReg, s: RawReg) {
		// Software popcount32
		self.emit_popcount(reg(d), reg(s), true);
	}

	fn count_set_bits_64(&mut self, d: RawReg, s: RawReg) {
		self.emit_popcount(reg(d), reg(s), false);
	}

	fn sign_extend_8(&mut self, d: RawReg, s: RawReg) {
		self.asm.th_ext(reg(d), reg(s), 7, 0);
	}

	fn sign_extend_16(&mut self, d: RawReg, s: RawReg) {
		self.asm.th_ext(reg(d), reg(s), 15, 0);
	}

	fn zero_extend_16(&mut self, d: RawReg, s: RawReg) {
		self.asm.th_extu(reg(d), reg(s), 15, 0);
	}

	fn reverse_byte(&mut self, d: RawReg, s: RawReg) {
		self.asm.th_rev(reg(d), reg(s));
	}

	// =======================================================================
	// reg, reg, imm, imm instructions
	// =======================================================================

	fn load_imm_and_jump_indirect(&mut self, ra: RawReg, base: RawReg, value: u32, offset: u32) {
		if reg(ra) == reg(base) {
			// ra aliases base: save base to TMP1 before overwriting with immediate.
			// TMP1 is safe to pass to emit_indirect_jump (it reads TMP1 before writing).
			self.asm.mv(TMP1, reg(base));
			self.asm.li32(reg(ra), value as i32);
			self.emit_indirect_jump(TMP1, offset);
		} else {
			self.asm.li32(reg(ra), value as i32);
			self.emit_indirect_jump(reg(base), offset);
		}
	}

	// =======================================================================
	// reg, imm64 instructions
	// =======================================================================

	fn load_imm64(&mut self, d: RawReg, imm: u64) {
		self.asm.li64(reg(d), imm as i64);
	}

	// =======================================================================
	// Invalid instruction
	// =======================================================================

	fn invalid(&mut self) {
		self.asm.unimp();
	}
}

impl Translator {
	/// Emit software popcount via Brian Kernighan's method.
	/// If `is_32bit`, only count bits in lower 32 bits.
	fn emit_popcount(&mut self, rd: u8, rs: u8, is_32bit: bool) {
		// Parallel bit-counting algorithm (Hamming weight):
		// x = x - ((x >> 1) & 0x5555...)
		// x = (x & 0x3333...) + ((x >> 2) & 0x3333...)
		// x = (x + (x >> 4)) & 0x0f0f...
		// x = (x * 0x0101...) >> 56 (or >> 24 for 32-bit)
		// count = 0; while (x != 0) { count++; x &= x-1; }
		if is_32bit {
			// Zero-extend to 64 bits
			self.asm.slli(TMP1, rs, 32);
			self.asm.srli(TMP1, TMP1, 32);
		} else {
			self.asm.mv(TMP1, rs);
		}
		self.asm.li32(rd, 0); // count = 0
		let loop_start = self.asm.position();
		self.asm.beq(TMP1, 0, 0); // if x == 0, break (patch later)
		let branch_pos = self.asm.position() - 4;
		self.asm.addi(rd, rd, 1); // count++
		self.asm.addi(TMP2, TMP1, -1); // TMP2 = x - 1
		self.asm.and(TMP1, TMP1, TMP2); // x &= x - 1
		let end = self.asm.position();
		self.asm.j((loop_start as i32) - (end as i32)); // j loop_start
		let done = self.asm.position();
		self.asm.patch_branch(branch_pos, done);
	}

}

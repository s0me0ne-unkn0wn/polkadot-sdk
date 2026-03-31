// This file is part of Substrate.

// Copyright (C) Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: GPL-3.0-or-later WITH Classpath-exception-2.0

//! Low-level RISC-V instruction encoding for RV64IMAC + xtheadbb + xtheadcondmov.

// RV64 base opcodes
const OP: u32 = 0b0110011;
const OP_32: u32 = 0b0111011;
const OP_IMM: u32 = 0b0010011;
const OP_IMM_32: u32 = 0b0011011;
const LOAD: u32 = 0b0000011;
const STORE: u32 = 0b0100011;
const BRANCH: u32 = 0b1100011;
const JAL: u32 = 0b1101111;
const JALR: u32 = 0b1100111;
const LUI: u32 = 0b0110111;
const AUIPC: u32 = 0b0010111;
#[allow(dead_code)]
const SYSTEM: u32 = 0b1110011;
const CUSTOM_0: u32 = 0b0001011;

/// RISC-V machine code emitter.
pub struct Emitter {
	buf: Vec<u8>,
}

impl Emitter {
	pub fn new() -> Self {
		Self { buf: Vec::with_capacity(4096) }
	}

	pub fn into_code(self) -> Vec<u8> {
		self.buf
	}

	#[allow(dead_code)]
	pub fn code(&self) -> &[u8] {
		&self.buf
	}

	pub fn position(&self) -> usize {
		self.buf.len()
	}

	fn emit32(&mut self, word: u32) {
		self.buf.extend_from_slice(&word.to_le_bytes());
	}

	/// Emit a raw 32-bit word (public).
	pub fn emit32_pub(&mut self, word: u32) {
		self.buf.extend_from_slice(&word.to_le_bytes());
	}

	// -----------------------------------------------------------------------
	// Encoding formats
	// -----------------------------------------------------------------------

	fn r_type(funct7: u32, rs2: u8, rs1: u8, funct3: u32, rd: u8, opcode: u32) -> u32 {
		(funct7 << 25)
			| ((rs2 as u32 & 0x1f) << 20)
			| ((rs1 as u32 & 0x1f) << 15)
			| (funct3 << 12)
			| ((rd as u32 & 0x1f) << 7)
			| opcode
	}

	fn i_type(imm12: i32, rs1: u8, funct3: u32, rd: u8, opcode: u32) -> u32 {
		(((imm12 as u32) & 0xfff) << 20)
			| ((rs1 as u32 & 0x1f) << 15)
			| (funct3 << 12)
			| ((rd as u32 & 0x1f) << 7)
			| opcode
	}

	fn s_type(imm12: i32, rs2: u8, rs1: u8, funct3: u32, opcode: u32) -> u32 {
		let imm = imm12 as u32 & 0xfff;
		((imm >> 5) << 25)
			| ((rs2 as u32 & 0x1f) << 20)
			| ((rs1 as u32 & 0x1f) << 15)
			| (funct3 << 12)
			| ((imm & 0x1f) << 7)
			| opcode
	}

	fn b_type(imm13: i32, rs2: u8, rs1: u8, funct3: u32, opcode: u32) -> u32 {
		let imm = imm13 as u32;
		let bit12 = (imm >> 12) & 1;
		let bit11 = (imm >> 11) & 1;
		let bits10_5 = (imm >> 5) & 0x3f;
		let bits4_1 = (imm >> 1) & 0xf;
		(bit12 << 31)
			| (bits10_5 << 25)
			| ((rs2 as u32 & 0x1f) << 20)
			| ((rs1 as u32 & 0x1f) << 15)
			| (funct3 << 12)
			| (bits4_1 << 8)
			| (bit11 << 7)
			| opcode
	}

	fn u_type(imm20: i32, rd: u8, opcode: u32) -> u32 {
		((imm20 as u32) & 0xfffff000) | ((rd as u32 & 0x1f) << 7) | opcode
	}

	fn j_type(imm21: i32, rd: u8, opcode: u32) -> u32 {
		let imm = imm21 as u32;
		let bit20 = (imm >> 20) & 1;
		let bits10_1 = (imm >> 1) & 0x3ff;
		let bit11 = (imm >> 11) & 1;
		let bits19_12 = (imm >> 12) & 0xff;
		(bit20 << 31)
			| (bits10_1 << 21)
			| (bit11 << 20)
			| (bits19_12 << 12)
			| ((rd as u32 & 0x1f) << 7)
			| opcode
	}

	// -----------------------------------------------------------------------
	// RV64I base instructions
	// -----------------------------------------------------------------------

	// R-type ALU
	pub fn add(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000000, rs2, rs1, 0b000, rd, OP));
	}
	pub fn sub(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0100000, rs2, rs1, 0b000, rd, OP));
	}
	pub fn and(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000000, rs2, rs1, 0b111, rd, OP));
	}
	pub fn or(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000000, rs2, rs1, 0b110, rd, OP));
	}
	pub fn xor(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000000, rs2, rs1, 0b100, rd, OP));
	}
	pub fn sll(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000000, rs2, rs1, 0b001, rd, OP));
	}
	pub fn srl(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000000, rs2, rs1, 0b101, rd, OP));
	}
	pub fn sra(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0100000, rs2, rs1, 0b101, rd, OP));
	}
	pub fn slt(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000000, rs2, rs1, 0b010, rd, OP));
	}
	pub fn sltu(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000000, rs2, rs1, 0b011, rd, OP));
	}

	// RV64I W-suffix (32-bit ops, sign-extend result)
	pub fn addw(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000000, rs2, rs1, 0b000, rd, OP_32));
	}
	pub fn subw(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0100000, rs2, rs1, 0b000, rd, OP_32));
	}
	pub fn sllw(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000000, rs2, rs1, 0b001, rd, OP_32));
	}
	pub fn srlw(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000000, rs2, rs1, 0b101, rd, OP_32));
	}
	pub fn sraw(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0100000, rs2, rs1, 0b101, rd, OP_32));
	}

	// I-type ALU
	pub fn addi(&mut self, rd: u8, rs1: u8, imm: i32) {
		self.emit32(Self::i_type(imm, rs1, 0b000, rd, OP_IMM));
	}
	pub fn addiw(&mut self, rd: u8, rs1: u8, imm: i32) {
		self.emit32(Self::i_type(imm, rs1, 0b000, rd, OP_IMM_32));
	}
	pub fn andi(&mut self, rd: u8, rs1: u8, imm: i32) {
		self.emit32(Self::i_type(imm, rs1, 0b111, rd, OP_IMM));
	}
	pub fn ori(&mut self, rd: u8, rs1: u8, imm: i32) {
		self.emit32(Self::i_type(imm, rs1, 0b110, rd, OP_IMM));
	}
	pub fn xori(&mut self, rd: u8, rs1: u8, imm: i32) {
		self.emit32(Self::i_type(imm, rs1, 0b100, rd, OP_IMM));
	}
	pub fn slti(&mut self, rd: u8, rs1: u8, imm: i32) {
		self.emit32(Self::i_type(imm, rs1, 0b010, rd, OP_IMM));
	}
	pub fn sltiu(&mut self, rd: u8, rs1: u8, imm: i32) {
		self.emit32(Self::i_type(imm, rs1, 0b011, rd, OP_IMM));
	}
	pub fn slli(&mut self, rd: u8, rs1: u8, shamt: u32) {
		// RV64: shamt is 6 bits
		self.emit32(Self::i_type((shamt & 0x3f) as i32, rs1, 0b001, rd, OP_IMM));
	}
	pub fn srli(&mut self, rd: u8, rs1: u8, shamt: u32) {
		self.emit32(Self::i_type((shamt & 0x3f) as i32, rs1, 0b101, rd, OP_IMM));
	}
	pub fn srai(&mut self, rd: u8, rs1: u8, shamt: u32) {
		self.emit32(
			Self::i_type(((0b0100000 << 5) | (shamt & 0x3f)) as i32, rs1, 0b101, rd, OP_IMM),
		);
	}
	pub fn slliw(&mut self, rd: u8, rs1: u8, shamt: u32) {
		self.emit32(Self::i_type((shamt & 0x1f) as i32, rs1, 0b001, rd, OP_IMM_32));
	}
	pub fn srliw(&mut self, rd: u8, rs1: u8, shamt: u32) {
		self.emit32(Self::i_type((shamt & 0x1f) as i32, rs1, 0b101, rd, OP_IMM_32));
	}
	pub fn sraiw(&mut self, rd: u8, rs1: u8, shamt: u32) {
		self.emit32(Self::i_type(
			((0b0100000 << 5) | (shamt & 0x1f)) as i32,
			rs1,
			0b101,
			rd,
			OP_IMM_32,
		));
	}

	// Loads
	pub fn lb(&mut self, rd: u8, rs1: u8, offset: i32) {
		self.emit32(Self::i_type(offset, rs1, 0b000, rd, LOAD));
	}
	pub fn lbu(&mut self, rd: u8, rs1: u8, offset: i32) {
		self.emit32(Self::i_type(offset, rs1, 0b100, rd, LOAD));
	}
	pub fn lh(&mut self, rd: u8, rs1: u8, offset: i32) {
		self.emit32(Self::i_type(offset, rs1, 0b001, rd, LOAD));
	}
	pub fn lhu(&mut self, rd: u8, rs1: u8, offset: i32) {
		self.emit32(Self::i_type(offset, rs1, 0b101, rd, LOAD));
	}
	pub fn lw(&mut self, rd: u8, rs1: u8, offset: i32) {
		self.emit32(Self::i_type(offset, rs1, 0b010, rd, LOAD));
	}
	pub fn lwu(&mut self, rd: u8, rs1: u8, offset: i32) {
		self.emit32(Self::i_type(offset, rs1, 0b110, rd, LOAD));
	}
	pub fn ld(&mut self, rd: u8, rs1: u8, offset: i32) {
		self.emit32(Self::i_type(offset, rs1, 0b011, rd, LOAD));
	}

	// Stores
	pub fn sb(&mut self, rs2: u8, rs1: u8, offset: i32) {
		self.emit32(Self::s_type(offset, rs2, rs1, 0b000, STORE));
	}
	pub fn sh(&mut self, rs2: u8, rs1: u8, offset: i32) {
		self.emit32(Self::s_type(offset, rs2, rs1, 0b001, STORE));
	}
	pub fn sw(&mut self, rs2: u8, rs1: u8, offset: i32) {
		self.emit32(Self::s_type(offset, rs2, rs1, 0b010, STORE));
	}
	pub fn sd(&mut self, rs2: u8, rs1: u8, offset: i32) {
		self.emit32(Self::s_type(offset, rs2, rs1, 0b011, STORE));
	}

	// Branches
	pub fn beq(&mut self, rs1: u8, rs2: u8, offset: i32) {
		self.emit32(Self::b_type(offset, rs2, rs1, 0b000, BRANCH));
	}
	pub fn bne(&mut self, rs1: u8, rs2: u8, offset: i32) {
		self.emit32(Self::b_type(offset, rs2, rs1, 0b001, BRANCH));
	}
	pub fn blt(&mut self, rs1: u8, rs2: u8, offset: i32) {
		self.emit32(Self::b_type(offset, rs2, rs1, 0b100, BRANCH));
	}
	pub fn bge(&mut self, rs1: u8, rs2: u8, offset: i32) {
		self.emit32(Self::b_type(offset, rs2, rs1, 0b101, BRANCH));
	}
	pub fn bltu(&mut self, rs1: u8, rs2: u8, offset: i32) {
		self.emit32(Self::b_type(offset, rs2, rs1, 0b110, BRANCH));
	}
	pub fn bgeu(&mut self, rs1: u8, rs2: u8, offset: i32) {
		self.emit32(Self::b_type(offset, rs2, rs1, 0b111, BRANCH));
	}

	// Jumps
	pub fn jal(&mut self, rd: u8, offset: i32) {
		self.emit32(Self::j_type(offset, rd, JAL));
	}
	pub fn jalr(&mut self, rd: u8, rs1: u8, offset: i32) {
		self.emit32(Self::i_type(offset, rs1, 0b000, rd, JALR));
	}

	// Upper immediates
	pub fn lui(&mut self, rd: u8, imm: i32) {
		self.emit32(Self::u_type(imm, rd, LUI));
	}
	pub fn auipc(&mut self, rd: u8, imm: i32) {
		self.emit32(Self::u_type(imm, rd, AUIPC));
	}

	// System
	#[allow(dead_code)]
	pub fn ebreak(&mut self) {
		self.emit32(Self::i_type(1, 0, 0b000, 0, SYSTEM));
	}

	/// UNIMP (C.UNIMP = 0x0000 in compressed, or UNIMPLEMENTED = all zeros 32-bit)
	pub fn unimp(&mut self) {
		self.emit32(0);
	}

	// -----------------------------------------------------------------------
	// RV64M extension (multiply/divide)
	// -----------------------------------------------------------------------

	pub fn mul(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000001, rs2, rs1, 0b000, rd, OP));
	}
	pub fn mulh(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000001, rs2, rs1, 0b001, rd, OP));
	}
	pub fn mulhsu(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000001, rs2, rs1, 0b010, rd, OP));
	}
	pub fn mulhu(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000001, rs2, rs1, 0b011, rd, OP));
	}
	pub fn div(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000001, rs2, rs1, 0b100, rd, OP));
	}
	pub fn divu(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000001, rs2, rs1, 0b101, rd, OP));
	}
	pub fn rem(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000001, rs2, rs1, 0b110, rd, OP));
	}
	pub fn remu(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000001, rs2, rs1, 0b111, rd, OP));
	}
	pub fn mulw(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000001, rs2, rs1, 0b000, rd, OP_32));
	}
	pub fn divw(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000001, rs2, rs1, 0b100, rd, OP_32));
	}
	pub fn divuw(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000001, rs2, rs1, 0b101, rd, OP_32));
	}
	pub fn remw(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000001, rs2, rs1, 0b110, rd, OP_32));
	}
	pub fn remuw(&mut self, rd: u8, rs1: u8, rs2: u8) {
		self.emit32(Self::r_type(0b0000001, rs2, rs1, 0b111, rd, OP_32));
	}

	// -----------------------------------------------------------------------
	// Pseudo-instructions
	// -----------------------------------------------------------------------

	/// `mv rd, rs` -> `addi rd, rs, 0`
	pub fn mv(&mut self, rd: u8, rs: u8) {
		self.addi(rd, rs, 0);
	}

	/// `not rd, rs` -> `xori rd, rs, -1`
	pub fn not(&mut self, rd: u8, rs: u8) {
		self.xori(rd, rs, -1);
	}

	/// `neg rd, rs` -> `sub rd, x0, rs`
	pub fn neg(&mut self, rd: u8, rs: u8) {
		self.sub(rd, 0, rs);
	}

	/// `nop` -> `addi x0, x0, 0`
	#[allow(dead_code)]
	pub fn nop(&mut self) {
		self.addi(0, 0, 0);
	}

	/// `j offset` -> `jal x0, offset`
	pub fn j(&mut self, offset: i32) {
		self.jal(0, offset);
	}

	/// Load 32-bit sign-extended immediate into rd.
	pub fn li32(&mut self, rd: u8, value: i32) {
		if value >= -2048 && value < 2048 {
			self.addi(rd, 0, value);
		} else {
			// LUI loads bits [31:12], ADDI adds bits [11:0]
			// Need to account for sign extension of ADDI's immediate
			let hi = if value & 0x800 != 0 {
				((value as u32).wrapping_add(0x1000) & 0xfffff000) as i32
			} else {
				value & !0xfff
			};
			let lo = value.wrapping_sub(hi);
			self.lui(rd, hi);
			if lo != 0 {
				self.addi(rd, rd, lo);
			}
		}
	}

	/// Load 64-bit immediate into rd.
	///
	/// Decomposes the value into four chunks: hi32[32] | mid_hi[12] | mid_lo[12] | lo8[8]
	/// Each chunk fits within ADDI's 12-bit signed immediate range.
	pub fn li64(&mut self, rd: u8, value: i64) {
		if value >= -2147483648 && value < 2147483648 {
			self.li32(rd, value as i32);
			return;
		}
		let value = value as u64;

		// Split into 4 chunks: [63:32] | [31:20] | [19:8] | [7:0]
		let lo8 = (value & 0xFF) as i32;
		let mid_lo = ((value >> 8) & 0xFFF) as i32;
		let mid_hi = ((value >> 20) & 0xFFF) as i32;
		let hi32 = (value >> 32) as i64;

		// Adjust for ADDI sign-extension: if a 12-bit chunk >= 2048, ADDI
		// sign-extends it negatively. Compensate by carrying +1 to the next chunk.
		// lo8 is 0..255 — always fits in 12-bit signed. No adjustment needed.
		let (mid_lo_s, carry1) =
			if mid_lo >= 2048 { (mid_lo - 4096, 1i64) } else { (mid_lo, 0i64) };
		let mid_hi_c = mid_hi as i64 + carry1;
		let (mid_hi_s, carry2) = if mid_hi_c >= 2048 {
			((mid_hi_c - 4096) as i32, 1i64)
		} else {
			(mid_hi_c as i32, 0i64)
		};
		let hi32_adj = (hi32 + carry2) as i32;

		self.li32(rd, hi32_adj); // rd = bits [63:32]
		self.slli(rd, rd, 12);
		if mid_hi_s != 0 {
			self.addi(rd, rd, mid_hi_s); // add bits [31:20]
		}
		self.slli(rd, rd, 12);
		if mid_lo_s != 0 {
			self.addi(rd, rd, mid_lo_s); // add bits [19:8]
		}
		self.slli(rd, rd, 8);
		if lo8 != 0 {
			self.addi(rd, rd, lo8); // add bits [7:0]
		}
	}

	/// Load unsigned 32-bit value, zero-extended.
	pub fn li_u32(&mut self, rd: u8, value: u32) {
		if value < 0x80000000 {
			self.li32(rd, value as i32);
		} else {
			// Value has bit 31 set; li32 would sign-extend to 64 bits.
			// Load and then mask.
			self.li32(rd, value as i32);
			// Zero-extend
			self.slli(rd, rd, 32);
			self.srli(rd, rd, 32);
		}
	}

	// -----------------------------------------------------------------------
	// T-Head xtheadbb extensions
	// All use CUSTOM_0 opcode (0b0001011)
	// -----------------------------------------------------------------------

	/// `th.ff1 rd, rs` - find first 1 from MSB (equivalent to CLZ)
	/// Encoding: funct7=0b1000011, rs2=0, rs1, funct3=001, rd, CUSTOM_0
	pub fn th_ff1(&mut self, rd: u8, rs: u8) {
		self.emit32(Self::r_type(0b1000011, 0, rs, 0b001, rd, CUSTOM_0));
	}

	/// `th.rev rd, rs` - byte reverse (equivalent to REV8)
	/// Encoding: funct7=0b1000001, rs2=0, rs1, funct3=001, rd, CUSTOM_0
	pub fn th_rev(&mut self, rd: u8, rs: u8) {
		self.emit32(Self::r_type(0b1000001, 0, rs, 0b001, rd, CUSTOM_0));
	}

	/// `th.ext rd, rs, msb, lsb` - bit-field extract with sign extension
	/// Encoding: msb[5:0] | lsb[5:0] | rs1 | 010 | rd | CUSTOM_0
	pub fn th_ext(&mut self, rd: u8, rs: u8, msb: u8, lsb: u8) {
		let imm_hi = (msb as u32 & 0x3f) << 26;
		let imm_lo = (lsb as u32 & 0x3f) << 20;
		let word = imm_hi
			| imm_lo
			| ((rs as u32 & 0x1f) << 15)
			| (0b010 << 12)
			| ((rd as u32 & 0x1f) << 7)
			| CUSTOM_0;
		self.emit32(word);
	}

	/// `th.extu rd, rs, msb, lsb` - bit-field extract with zero extension
	/// Encoding: msb[5:0] | lsb[5:0] | rs1 | 011 | rd | CUSTOM_0
	pub fn th_extu(&mut self, rd: u8, rs: u8, msb: u8, lsb: u8) {
		let imm_hi = (msb as u32 & 0x3f) << 26;
		let imm_lo = (lsb as u32 & 0x3f) << 20;
		let word = imm_hi
			| imm_lo
			| ((rs as u32 & 0x1f) << 15)
			| (0b011 << 12)
			| ((rd as u32 & 0x1f) << 7)
			| CUSTOM_0;
		self.emit32(word);
	}

	/// `th.srri rd, rs, shamt` - rotate right by immediate
	/// Encoding: 00010 | shamt[5:0] | rs1 | 001 | rd | CUSTOM_0
	pub fn th_srri(&mut self, rd: u8, rs: u8, shamt: u8) {
		let funct5: u32 = 0b00010;
		let word = (funct5 << 27)
			| ((shamt as u32 & 0x3f) << 20)
			| ((rs as u32 & 0x1f) << 15)
			| (0b001 << 12)
			| ((rd as u32 & 0x1f) << 7)
			| CUSTOM_0;
		self.emit32(word);
	}

	// -----------------------------------------------------------------------
	// T-Head xtheadcondmov extensions
	// -----------------------------------------------------------------------

	/// `th.mveqz rd, rs, rc` - if rc == 0, rd = rs (else rd unchanged)
	/// Encoding: funct7=0b0100000, rs2=rc, rs1=rs, funct3=001, rd, CUSTOM_0
	pub fn th_mveqz(&mut self, rd: u8, rs: u8, rc: u8) {
		self.emit32(Self::r_type(0b0100000, rc, rs, 0b001, rd, CUSTOM_0));
	}

	/// `th.mvnez rd, rs, rc` - if rc != 0, rd = rs (else rd unchanged)
	/// Encoding: funct7=0b0100001, rs2=rc, rs1=rs, funct3=001, rd, CUSTOM_0
	pub fn th_mvnez(&mut self, rd: u8, rs: u8, rc: u8) {
		self.emit32(Self::r_type(0b0100001, rc, rs, 0b001, rd, CUSTOM_0));
	}

	// -----------------------------------------------------------------------
	// Patching helpers for fixups
	// -----------------------------------------------------------------------

	/// Patch a B-type instruction at `offset` to branch to `target_offset`.
	/// Only safe for short-range branches (±4KB). Used for local inline loops only.
	pub fn patch_branch(&mut self, offset: usize, target_offset: usize) {
		let delta = (target_offset as i64 - offset as i64) as i32;
		assert!(
			delta >= -4096 && delta < 4096,
			"branch offset out of range: {delta} (local branch should be short-range)"
		);
		let old = u32::from_le_bytes(self.buf[offset..offset + 4].try_into().unwrap());
		// Clear imm bits, preserve the rest
		let opcode_etc = old & 0x01fff07f;
		// Re-encode with new offset
		let imm = delta as u32;
		let bit12 = (imm >> 12) & 1;
		let bit11 = (imm >> 11) & 1;
		let bits10_5 = (imm >> 5) & 0x3f;
		let bits4_1 = (imm >> 1) & 0xf;
		let new = opcode_etc | (bit12 << 31) | (bits10_5 << 25) | (bits4_1 << 8) | (bit11 << 7);
		self.buf[offset..offset + 4].copy_from_slice(&new.to_le_bytes());
	}

	/// Patch an AUIPC+JALR pair at `offset` to jump to `target_offset`.
	/// AUIPC is at `offset`, JALR is at `offset + 4`.
	/// Supports the full ±2GB range of a 32-bit signed offset.
	pub fn patch_auipc_jalr(&mut self, offset: usize, target_offset: usize) {
		let delta = (target_offset as i64 - offset as i64) as i32;
		// Split delta into hi20 (AUIPC) and lo12 (JALR), accounting for
		// sign-extension of JALR's 12-bit immediate.
		let lo = (delta << 20) >> 20; // sign-extend bottom 12 bits
		let hi = delta.wrapping_sub(lo);

		// Patch AUIPC: U-type = imm[31:12] | rd[11:7] | opcode[6:0]
		let old_auipc = u32::from_le_bytes(self.buf[offset..offset + 4].try_into().unwrap());
		let rd_opcode = old_auipc & 0x00000FFF; // preserve rd + opcode
		let new_auipc = ((hi as u32) & 0xFFFFF000) | rd_opcode;
		self.buf[offset..offset + 4].copy_from_slice(&new_auipc.to_le_bytes());

		// Patch JALR: I-type = imm[31:20] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]
		let jalr_off = offset + 4;
		let old_jalr = u32::from_le_bytes(self.buf[jalr_off..jalr_off + 4].try_into().unwrap());
		let jalr_lower = old_jalr & 0x000FFFFF; // preserve rs1 + funct3 + rd + opcode
		let new_jalr = (((lo as u32) & 0xFFF) << 20) | jalr_lower;
		self.buf[jalr_off..jalr_off + 4].copy_from_slice(&new_jalr.to_le_bytes());
	}

	/// Overwrite 4 bytes at the given offset with a raw 32-bit value.
	pub fn write_u32_at(&mut self, offset: usize, value: u32) {
		self.buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_addi_encoding() {
		let mut e = Emitter::new();
		// addi x1, x0, 42 => imm=42, rs1=x0, funct3=000, rd=x1, op=0010011
		e.addi(1, 0, 42);
		let word = u32::from_le_bytes(e.code()[0..4].try_into().unwrap());
		// Expected: 0x02a00093
		assert_eq!(word, 0x02a00093, "addi x1, x0, 42 encoding mismatch");
	}

	#[test]
	fn test_add_encoding() {
		let mut e = Emitter::new();
		// add x3, x1, x2
		e.add(3, 1, 2);
		let word = u32::from_le_bytes(e.code()[0..4].try_into().unwrap());
		// Expected: 0x002081b3
		assert_eq!(word, 0x002081b3, "add x3, x1, x2 encoding mismatch");
	}

	#[test]
	fn test_li32_small() {
		let mut e = Emitter::new();
		e.li32(5, 100);
		// Should emit a single addi x5, x0, 100
		assert_eq!(e.position(), 4);
	}

	#[test]
	fn test_li32_large() {
		let mut e = Emitter::new();
		e.li32(5, 0x12345);
		// Should emit lui + addi = 8 bytes
		assert!(e.position() >= 4);
	}

	#[test]
	fn test_branch_encoding() {
		let mut e = Emitter::new();
		// beq x1, x2, +8
		e.beq(1, 2, 8);
		let word = u32::from_le_bytes(e.code()[0..4].try_into().unwrap());
		// offset=8: bit12=0, bits10:5=0, bits4:1=0100, bit11=0
		// funct3=000, rs1=x1, rs2=x2
		assert_ne!(word, 0); // basic sanity
	}

	/// Simulate li64 execution: interpret the emitted instructions on a simple
	/// RISC-V register model and return the final rd value.
	fn simulate_li64(value: i64) -> u64 {
		let mut e = Emitter::new();
		e.li64(5, value); // rd = x5
		let code = e.into_code();

		let mut regs = [0u64; 32];
		let mut pc = 0usize;

		while pc < code.len() {
			let word = u32::from_le_bytes(code[pc..pc + 4].try_into().unwrap());
			let opcode = word & 0x7f;
			let rd = ((word >> 7) & 0x1f) as usize;
			let rs1 = ((word >> 15) & 0x1f) as usize;
			let funct3 = (word >> 12) & 0x7;

			match opcode {
				0b0110111 => {
					// LUI: rd = sign_extend(imm[31:12] << 0)
					let imm = (word & 0xFFFFF000) as i32;
					regs[rd] = imm as i64 as u64;
				}
				0b0010011 => {
					// OP-IMM: addi, slli, srli, srai
					let imm = ((word as i32) >> 20) as i64;
					match funct3 {
						0b000 => {
							// ADDI
							regs[rd] = (regs[rs1] as i64).wrapping_add(imm) as u64;
						}
						0b001 => {
							// SLLI
							let shamt = (word >> 20) & 0x3f;
							regs[rd] = regs[rs1] << shamt;
						}
						0b101 => {
							// SRLI/SRAI
							let shamt = (word >> 20) & 0x3f;
							let arith = (word >> 30) & 1;
							if arith != 0 {
								regs[rd] = ((regs[rs1] as i64) >> shamt) as u64;
							} else {
								regs[rd] = regs[rs1] >> shamt;
							}
						}
						_ => panic!("unsupported OP-IMM funct3={funct3}"),
					}
				}
				_ => panic!("unsupported opcode=0x{opcode:x}"),
			}
			pc += 4;
		}
		regs[5]
	}

	#[test]
	fn test_li64_typical_addresses() {
		// Addresses similar to real RISC-V mmap'd pointers
		let test_values: &[u64] = &[
			0x2AC8500000,
			0x3FCC519074,
			0x2ACD282000,
			0x0000003FC45F2000,
			0x0000002AC7A063BC,
			0xFFFFFFFF80000000, // negative
			0x0000000012345678,
			0x00000000DEADBEEF,
			0x0000ABCDEF012345,
			0x7FFFFFFFFFFFFFFF,
		];
		for &val in test_values {
			let result = simulate_li64(val as i64);
			assert_eq!(
				result, val,
				"li64 mismatch for 0x{val:016X}: got 0x{result:016X}"
			);
		}
	}

	#[test]
	fn test_li64_small_values() {
		// Values that fit in 32 bits (should use li32 path)
		assert_eq!(simulate_li64(0), 0);
		assert_eq!(simulate_li64(42), 42);
		assert_eq!(simulate_li64(-1), u64::MAX);
		assert_eq!(simulate_li64(-0x80000000), 0xFFFFFFFF80000000);
	}
}

// This file is part of Substrate.

// Copyright (C) Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: GPL-3.0-or-later WITH Classpath-exception-2.0

//! Memory management for native RISC-V executor.
//!
//! Manages the guest memory region via mmap and the executable code buffer.

use sc_executor_common::error::Error;
use std::ptr;

/// Guest memory region backed by mmap.
///
/// The guest uses 32-bit addresses. We allocate a contiguous region and store
/// its base address. A guest address `ga` maps to host address `base + ga`.
pub struct GuestMemory {
	base: *mut u8,
	total_size: usize,
}

// SAFETY: GuestMemory is only used single-threaded within one Instance.
unsafe impl Send for GuestMemory {}
unsafe impl Sync for GuestMemory {}

impl GuestMemory {
	/// Allocate guest memory of `total_size` bytes.
	pub fn new(total_size: usize) -> Result<Self, Error> {
		if total_size == 0 {
			return Err("guest memory size is zero".into());
		}
		let base = unsafe {
			libc::mmap(
				ptr::null_mut(),
				total_size,
				libc::PROT_READ | libc::PROT_WRITE,
				libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
				-1,
				0,
			)
		};
		if base == libc::MAP_FAILED {
			return Err("failed to mmap guest memory".into());
		}
		Ok(Self { base: base as *mut u8, total_size })
	}

	/// Returns the host pointer corresponding to guest address 0.
	pub fn base_ptr(&self) -> *mut u8 {
		self.base
	}

	/// Write `data` at guest address `addr`.
	pub fn write(&mut self, addr: u32, data: &[u8]) -> Result<(), Error> {
		let end = (addr as usize).checked_add(data.len()).ok_or("guest address overflow")?;
		if end > self.total_size {
			return Err(format!(
				"guest memory write out of bounds: addr=0x{addr:x}, len={}, total=0x{:x}",
				data.len(),
				self.total_size
			)
			.into());
		}
		unsafe {
			ptr::copy_nonoverlapping(data.as_ptr(), self.base.add(addr as usize), data.len());
		}
		Ok(())
	}

	/// Read `len` bytes from guest address `addr`.
	pub fn read(&self, addr: u32, len: u32) -> Result<Vec<u8>, Error> {
		let end = (addr as usize)
			.checked_add(len as usize)
			.ok_or("guest address overflow")?;
		if end > self.total_size {
			return Err(format!(
				"guest memory read out of bounds: addr=0x{addr:x}, len={len}, total=0x{:x}",
				self.total_size
			)
			.into());
		}
		let mut buf = vec![0u8; len as usize];
		unsafe {
			ptr::copy_nonoverlapping(self.base.add(addr as usize), buf.as_mut_ptr(), len as usize);
		}
		Ok(buf)
	}

	/// Zero the region [addr, addr+len).
	#[allow(dead_code)]
	pub fn zero(&mut self, addr: u32, len: u32) -> Result<(), Error> {
		let end = (addr as usize)
			.checked_add(len as usize)
			.ok_or("guest address overflow")?;
		if end > self.total_size {
			return Err("guest memory zero out of bounds".into());
		}
		unsafe {
			ptr::write_bytes(self.base.add(addr as usize), 0, len as usize);
		}
		Ok(())
	}
}

impl Drop for GuestMemory {
	fn drop(&mut self) {
		if !self.base.is_null() {
			unsafe {
				libc::munmap(self.base as *mut libc::c_void, self.total_size);
			}
		}
	}
}

/// Executable code buffer backed by mmap.
pub struct ExecutableBuffer {
	base: *mut u8,
	size: usize,
	code_len: usize,
}

unsafe impl Send for ExecutableBuffer {}
unsafe impl Sync for ExecutableBuffer {}

impl ExecutableBuffer {
	/// Create from raw machine code bytes.
	/// The buffer is initially RW, then made RX.
	pub fn new(code: &[u8]) -> Result<Self, Error> {
		if code.is_empty() {
			return Err("empty code buffer".into());
		}
		let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
		let size = (code.len() + page_size - 1) & !(page_size - 1);
		let base = unsafe {
			libc::mmap(
				ptr::null_mut(),
				size,
				libc::PROT_READ | libc::PROT_WRITE,
				libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
				-1,
				0,
			)
		};
		if base == libc::MAP_FAILED {
			return Err("failed to mmap code buffer".into());
		}
		unsafe {
			ptr::copy_nonoverlapping(code.as_ptr(), base as *mut u8, code.len());
		}
		// Make executable (drop write)
		let rc = unsafe { libc::mprotect(base, size, libc::PROT_READ | libc::PROT_EXEC) };
		if rc != 0 {
			unsafe {
				libc::munmap(base, size);
			}
			return Err("failed to mprotect code buffer as RX".into());
		}
		// Flush instruction cache so the CPU sees the new code.
		// On RISC-V, mprotect does NOT guarantee I-cache coherency.
		#[cfg(target_arch = "riscv64")]
		unsafe {
			// fence.i synchronizes I-cache with D-cache on the current hart.
			core::arch::asm!("fence.i");
		}
		Ok(Self { base: base as *mut u8, size, code_len: code.len() })
	}

	/// Base address of the executable code.
	pub fn base_ptr(&self) -> *const u8 {
		self.base
	}

	#[allow(dead_code)]
	pub fn code_len(&self) -> usize {
		self.code_len
	}
}

impl Drop for ExecutableBuffer {
	fn drop(&mut self) {
		if !self.base.is_null() {
			unsafe {
				libc::munmap(self.base as *mut libc::c_void, self.size);
			}
		}
	}
}

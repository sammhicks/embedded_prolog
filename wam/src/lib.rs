#![allow(dead_code, unused_variables)]
#![feature(never_type)]
#![no_std]

use embedded_hal::serial;

mod io;

use io::{Header, SerialWriter};

enum State {
    WaitingForProgram,
    WaitingForQuery {
        program_size: usize,
    },
    ChoicePoint {
        program_size: usize,
        query_size: usize,
    },
}

fn handle_code_submission<SR, SW>(
    reader: &mut SR,
    writer: &mut SW,
    memory: &'static mut [u32],
) -> Result<usize, io::ReadError>
where
    SR: serial::Read<u8, Error = !>,
    SW: serial::Write<u8, Error = !>,
{
    let len = io::read_u32(reader)? as usize;

    for word in &mut memory[..len] {
        *word = io::read_u32(reader)?;
    }

    //TODO, read hash and check

    SerialWriter::send_str(writer, "S\n");

    Ok(len)
}

pub fn run_wam<SR, SW>(mut reader: SR, mut writer: SW, memory: &mut [u32]) -> !
where
    SR: serial::Read<u8, Error = !>,
    SW: serial::Write<u8, Error = !>,
{
    let mut state = State::WaitingForProgram;

    loop {
        let memory = unsafe { core::slice::from_raw_parts_mut(memory.as_mut_ptr(), memory.len()) };
        state = match state {
            State::WaitingForProgram => match io::read_header(&mut reader) {
                Ok(Header::ReportStatus) => {
                    SerialWriter::send_str(&mut writer, "P\n");
                    State::WaitingForProgram
                }
                Ok(Header::SubmitProgram) => {
                    match handle_code_submission(&mut reader, &mut writer, memory) {
                        Ok(program_size) => State::WaitingForQuery { program_size },
                        Err(e) => {
                            error!(&mut writer, "{:?}", e);
                            State::WaitingForProgram
                        }
                    }
                }
                Ok(Header::SubmitQuery) => {
                    error!(&mut writer, "Cannot Accept Query");
                    State::WaitingForProgram
                }
                Err(e) => {
                    error!(&mut writer, "Not a header: {:x}", e.number);
                    State::WaitingForProgram
                }
            },
            State::WaitingForQuery { program_size } => match io::read_header(&mut reader) {
                Ok(Header::ReportStatus) => {
                    SerialWriter::send_str(&mut writer, "Q\n");
                    State::WaitingForQuery { program_size }
                }
                Ok(Header::SubmitProgram) => {
                    match handle_code_submission(&mut reader, &mut writer, memory) {
                        Ok(program_size) => State::WaitingForQuery { program_size },
                        Err(e) => {
                            error!(&mut writer, "{:?}", e);
                            State::WaitingForProgram
                        }
                    }
                }
                Ok(Header::SubmitQuery) => {
                    match handle_code_submission(
                        &mut reader,
                        &mut writer,
                        &mut memory[program_size..],
                    ) {
                        Ok(query_size) => unimplemented!(),
                        Err(e) => {
                            error!(&mut writer, "{:?}", e);
                            State::WaitingForQuery { program_size }
                        }
                    }
                }
                Err(e) => {
                    error!(&mut writer, "Not a header: {:x}", e.number);
                    State::WaitingForQuery { program_size }
                }
            },
            _ => state,
        }
    }
}

use super::{
    log_trace, CommandHeader, ProcessInputError, SerialConnection, SerialRead, SerialWrite,
};

#[derive(Debug)]
pub enum UnhandledCommand {
    ProcessNextCommand,
    SubmitProgram,
    SubmitQuery,
}

pub struct Device<'a, 's1, 's2, S> {
    pub program: &'a [u32],
    pub query: &'a [u32],
    pub memory: &'a mut [u32],
    pub serial_connection: &'s1 mut &'s2 mut SerialConnection<S>,
}

impl<'a, 's1, 's2, S: SerialRead<u8> + SerialWrite<u8>> Device<'a, 's1, 's2, S> {
    pub fn run(self) -> Result<UnhandledCommand, ProcessInputError> {
        let (machine, success) = match super::machine::Machine::run(super::machine::MachineMemory {
            program: self.program,
            query: self.query,
            memory: self.memory,
        }) {
            Ok(success) => success,
            Err(failure) => {
                self.serial_connection
                    .write_single_char(failure as u8 as char)?;
                return Ok(UnhandledCommand::ProcessNextCommand);
            }
        };

        self.serial_connection.write_char(success as u8 as char)?;
        for address in machine.solution_registers() {
            log_trace!("Solution Register: {}", address);
            self.serial_connection.write_be_serializable_hex(address)?;
        }
        self.serial_connection.flush()?;

        loop {
            let command = CommandHeader::parse(self.serial_connection.read_ascii_char()?)?;

            log::debug!("Processing command {:?}", command);

            match command {
                CommandHeader::ReportStatus => {
                    log::debug!("Status: Waiting for Query");
                    self.serial_connection
                        .write_single_char(success as u8 as char)?;
                }
                CommandHeader::SubmitProgram => return Ok(UnhandledCommand::SubmitProgram),
                CommandHeader::SubmitQuery => return Ok(UnhandledCommand::SubmitQuery),
                CommandHeader::LookupMemory => {
                    let (address, value, subterms) =
                        machine.lookup_memory(self.serial_connection.read_be_serializable_hex()?);
                    self.serial_connection
                        .write_value(address, value, subterms)?;
                    self.serial_connection.flush()?;
                }
                CommandHeader::NextSolution => {
                    todo!()
                }
            }
        }
    }
}

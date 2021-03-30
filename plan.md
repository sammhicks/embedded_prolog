# Commands
+ S - Report Status
  + P - Waiting for Program
  + Q - Waiting for Query
  + R - Executing Program
  + A - Single Answer, then predicate values
  + C - Multiple Answers, then predicate values
  + F - No Answers
+ P - Submit Program
  + length in words(u32): BE u32 in hex
  + program words in BE hex
  + SHA256 hash in BE hex
  + Response on success: 'S'
+ Q - Run Query
  + length in words(u32): BE u32 in hex
  + query words in BE hex
  + SHA256 hash in BE hex
  + Response on success: Current Status
+ M - Lookup Memory, with memory index: BE u16 in hex
  + R - Reference, followed by address: BE u16 in hex
  + S - Structure, followed by:
    + Functor: BE u16 in hex
    + Arity: u8 in hex
    + List of elements: BE u32 in hex per item
  + L - List, followed by:
    + Head: u32 in hex
    + Tail: u32 in hex
  + C - Constant, followed by value: BE u16 in hex
  + I - Integer, followed by:
    + Word count: BE u8 in hex
    + List of Words: BE u32 in hex per word

# Error Responses
+ 'E' then bytes in hex then 'S'

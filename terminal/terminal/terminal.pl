:- module(terminal, [
              current_connection/1,
              disconnect/0,
              status/1,
              compile_program/2
          ]).

:- use_module(compiler/compiler).
:- use_module(io).
:- use_module(read_terms).

:- dynamic(current_connection/1).
:- dynamic(program_compile_state/1).

disconnect :-
    (   current_connection(Stream)
    ->  catch(close(Stream), Error, writeln(Error))
    ;   true),
    retractall(current_connection(_)).


status(Status) :-
    status(_Stream, Status).


status(Stream, Status) :-
    (   current_connection(Stream)
    ->  connected_status(Stream, Status)
    ;   Status = disconnected).


connected_status(Stream, Status) :-
    write(Stream, 'S'),
    flush_output(Stream),
    read_nonspace(Stream, Header),
    (   char_status(Header, Status)
    ->  read_error(Stream, Status)
    ;   Status = unknown_status(Header)
    ).


char_status('P', waiting_for_program).
char_status('Q', waiting_for_query).
char_status('R', executing_program).
char_status('A', single_answer).
char_status('C', multiple_answers).
char_status('F', no_answers).
char_status('E', error(_Message)).


read_error(Stream, Error) :-
    (   Error = error(Message)
    ->  read_error_chars(Stream, Chars),
        string_chars(Message, Chars)
    ;   true
    ).


read_error_chars(Stream, Chars) :-
    read_u32(Stream, Count),
    length(Section, Count),
    maplist(read_u8(Stream), Section),
    append(Section, Rest, Chars),
    read_nonspace(Stream, C),
    (   C == 'E'
    ->  read_error_chars(Stream, Rest)
    ;   C == 'S'
    ->  Rest = []
    ).


can_submit_program(waiting_for_program).
can_submit_program(waiting_for_query).
can_submit_program(single_answer).
can_submit_program(multiple_answers).
can_submit_program(no_answers).


compile_program(File, Result) :-
    status(Stream, Status),
    (   can_submit_program(Status)
    ->  read_terms_from_file(File, Program),
        retractall(program_compile_state(_)),
        compile_program(Program, State, Program_Words),
        write(Stream, 'P'),
        write_words(Stream, Program_Words),
        flush_output(Stream),
        read_upload_result(Stream, Result),
        assertz(program_compile_state(State))
    ;   Result = cannot_submit_program(Status)
    ).


read_upload_result(Stream, Result) :-
    read_nonspace(Stream, Header),
    (   char_upload_result(Header, Result)
    ->  read_error(Stream, Result)
    ;   Result = unknown_result(Header)
    ).


char_upload_result('S', success).
char_upload_result('E', error(_Message)).

:- module(terminal, [
              current_connection/1,
              disconnect/0,
              status/1,
              compile_program/2,
              run_query/2
          ]).

:- use_module(library(lists)).
:- use_module(library(ordsets)).
:- use_module(library(utf8)).

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
    ->  read_error_bytes(Stream, Bytes),
        utf8_codes(Codes, Bytes, []),
        string_codes(Message, Codes)
    ;   true
    ).


read_error_bytes(Stream, Bytes) :-
    try_read_u8(Stream, Result),
    (   Result = ok(B)
    ->  Bytes = [B|Tail],
        read_error_bytes(Stream, Tail)
    ;   Result = err('S'),
        Bytes = []
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
        write_words_with_hash(Stream, Program_Words),
        flush_output(Stream),
        read_program_result(Stream, Result),
        assertz(program_compile_state(State))
    ;   Result = cannot_submit_program(Status)
    ).


read_program_result(Stream, Result) :-
    read_nonspace(Stream, Header),
    (   program_upload_result(Header, Result)
    ->  read_error(Stream, Result)
    ;   Result = unknown_result(Header)
    ).


program_upload_result('S', success).
program_upload_result('E', error(_Message)).


can_submit_query(waiting_for_query).
can_submit_query(single_answer).
can_submit_query(multiple_answers).
can_submit_query(no_answers).


run_query(Query, Result) :-
    program_compile_state(State),
    status(Stream, Status),
    (   can_submit_query(Status)
    ->  compile_query(Query, State, Query_Words, Functors),
        write(Stream, 'Q'),
        write_words_with_hash(Stream, Query_Words),
        flush_output(Stream),
        read_query_result(Stream, Query, Functors, Result)
    ).


read_query_result(Stream, Query, Functors, Result) :-
    read_nonspace(Stream, Result_Header),
    handle_query_result(Result_Header, Stream, Query, Functors, Result).


handle_query_result('A', Stream, Query, Functors, single_answer) :-
    read_answer(Stream, Query, Functors).
handle_query_result('C', Stream, Query, Functors, multiple_answers) :-
    read_answer(Stream, Query, Functors).
handle_query_result('C', Stream, Query, Functors, Result) :-
    read_query_result(Stream, Query, Functors, Result).
handle_query_result('F', _Stream, _Query, _Functors, no_answer).
handle_query_result('E', Stream, _Query, _Functors, error(Message)) :-
    read_error(Stream, error(Message)).


read_answer(Stream, Query, Functors) :-
    compound_name_arity(Query, Name, Arity),
    length(Register_References, Arity),
    maplist(read_u16(Stream), Register_References),
    read_answer_references(Register_References, Stream, Functors, [], [], Lookup_Table),
    maplist(get_lookup_value(Lookup_Table), Register_References, Arguments),
    compound_name_arguments(Query, Name, Arguments).


read_answer_references([], _Stream, _Functors, _Resolved_References, Lookup_Table, Lookup_Table).
read_answer_references([Reference|References], Stream, Functors, Resolved_References, Current_Lookup_Table, Final_Lookup_Table) :-
    read_answer_reference(Stream, Functors, Reference, Resolved_References, New_References, Newly_Resolved_References, New_Entries),
    append(References, New_References, All_References),
    ord_union(Newly_Resolved_References, Resolved_References, Next_Resolved_References),
    insert_entries(New_Entries, Current_Lookup_Table, New_Lookup_Table),
    read_answer_references(All_References, Stream, Functors, Next_Resolved_References, New_Lookup_Table, Final_Lookup_Table).


read_answer_reference(_Stream, _Functors, Reference, Resolved_References, [], [], []) :-
    memberchk(Reference, Resolved_References),
    !.

read_answer_reference(Stream, Functors, Reference, _Resolved_References, New_References, Newly_Resolved_References, New_Entries) :-
    write(Stream, 'M'),
    write_u16(Stream, Reference),
    flush_output(Stream),
    read_nonspace(Stream, Value_Header),
    read_value(Value_Header, Stream, Value_Description),
    read_u16(Stream, Address),
    resolve_value_description(Value_Description, Functors, Value, New_References, Additional_Entries),
    list_to_ord_set([Reference, Address], Newly_Resolved_References),
    New_Entries = [Reference=Value, Address=Value|Additional_Entries].


resolve_value_description(reference(_R), _Functors, _Value, [], []).
resolve_value_description(structure(Functor, Arity, First_Term), Functors, Value, New_References, Additional_Entries) :-
    nth0(Functor, Functors, Name),
    length(Arguments, Arity),
    compound_name_arguments(Value, Name, Arguments),
    length(New_References, Arity),
    incrementing_series(New_References, First_Term),
    maplist(entry, New_References, Arguments, Additional_Entries),
    true.

resolve_value_description(constant(C), Functors, Value, [], []) :-
    nth0(C, Functors, Value).


read_value('R', Stream, reference(R)) :-
    read_u16(Stream, R).

read_value('S', Stream, structure(F, N, First_Term)) :-
    read_u16(Stream, F),
    read_u8(Stream, N),
    read_u16(Stream, First_Term).

read_value('C', Stream, constant(C)) :-
    read_u16(Stream, C).


incrementing_series([], _N).
incrementing_series([N|Ns], N) :-
    Successor is N + 1,
    incrementing_series(Ns, Successor).


insert_entries([], Table, Table).
insert_entries([Entry|Entries], Current_Table, Final_Table) :-
    insert_entry(Entry, Current_Table, New_Table),
    insert_entries(Entries, New_Table, Final_Table).


insert_entry(New_Entry, Table, Table) :-
    entry(Key, New_Value, New_Entry),
    entry(Key, Current_Value, Current_Entry),
    memberchk(Current_Entry, Table),
    !,
    Current_Value = New_Value.

insert_entry(New_Entry, Current_Table, [New_Entry|Current_Table]).



get_lookup_value(Lookup_Table, Reference, Value) :-
    entry(Reference, Value, Entry),
    memberchk(Entry, Lookup_Table).


entry(K, V, K=V).
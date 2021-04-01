
:- module(assembler, [
	      assemble_query/4,         % +Codes, +State, -Words, -Functors
	      assemble_program/3        % +Codes, -State, -Words
	  ]).

:- use_module(assembly_sections/allocate_labels).
:- use_module(assembly_sections/remove_labels).
:- use_module(assembly_sections/allocate_functors).

assemble_query(Codes0, State, Word_Components, Functors) :-
	assembly_state(State, Functors0, Labels),
	allocate_functors(Codes0, Codes1, Functors0, Functors),
	apply_labels(Codes1, Labels, Codes3),
	assemble_codes(Codes3, Word_Components_With_Program_Locations, []),
	assemble_program_locations(Word_Components_With_Program_Locations, Word_Components),
	maplist(is_valid_word, Word_Components).


assemble_program(Codes0, State, Word_Components) :-
	init_functors_state(Functors0),
	allocate_functors(Codes0, Codes1, Functors0, Functors),
	allocate_labels(Codes1, Labels),
	apply_labels(Codes1, Labels, Codes2),
	assemble_codes(Codes2, Word_Components_With_Labels, []),
	remove_labels(Word_Components_With_Labels, Word_Components_With_Program_Locations, Labels),
	assemble_program_locations(Word_Components_With_Program_Locations, Word_Components),
	assembly_state(State, Functors, Labels),
	maplist(is_valid_word, Word_Components).


assemble_codes([]) -->
	[].
assemble_codes([Code|Codes]) -->
	assemble_code(Code),
	!,
	assemble_codes(Codes).


assemble_code(put_variable(x(N), a(Ai))) --> [[0x00,   Ai, 0x00,    N]].
assemble_code(put_variable(y(N), a(Ai))) --> [[0x01,   Ai, 0x00,    N]].
assemble_code(put_value(x(N), a(Ai)))    --> [[0x02,   Ai, 0x00,    N]].
assemble_code(put_value(y(N), a(Ai)))    --> [[0x03,   Ai, 0x00,    N]].
assemble_code(put_structure(F/N, a(Ai))) --> [[0x04,   Ai,   F0,    N]], { uint(F, F0) }, !.
assemble_code(put_structure(F/N, a(Ai))) --> [[0x05,   Ai, 0x00,    N], [0x00, 0x00, F1, F0]], { uint(F, F1, F0) }.
assemble_code(put_list(a(Ai)))           --> [[0x06,   Ai, 0x00, 0x00]].
assemble_code(put_constant(C, a(Ai)))    --> [[0x07,   Ai,   C1,   C0]], { uint(C, C1, C0) }.
assemble_code(put_integer(I, a(Ai)))     --> [[0x08,   Ai,   I1,   I0]], { sint(I, I1, I0) }, !.
assemble_code(put_integer(I, a(Ai)))     --> [[0x09,   Ai,    S,    N]], wsint(I, S, N).
assemble_code(put_void(N, a(Ai)))        --> [[0x0a,   Ai, 0x00,    N]].

assemble_code(get_variable(x(N), a(Ai))) --> [[0x10,   Ai, 0x00,    N]].
assemble_code(get_variable(y(N), a(Ai))) --> [[0x11,   Ai, 0x00,    N]].
assemble_code(get_value(x(N), a(Ai)))    --> [[0x12,   Ai, 0x00,    N]].
assemble_code(get_value(y(N), a(Ai)))    --> [[0x13,   Ai, 0x00,    N]].
assemble_code(get_structure(F/N, a(Ai))) --> [[0x14,   Ai,    F,    N]].
assemble_code(get_structure(F/N, a(Ai))) --> [[0x15,   Ai, 0x00,    N], [0x00, 0x00, F1, F0]], { uint(F, F1, F0) }.
assemble_code(get_list(a(Ai)))           --> [[0x16,   Ai, 0x00, 0x00]].
assemble_code(get_constant(C, a(Ai)))    --> [[0x17,   Ai,   C1,   C0]], { uint(C, C1, C0) }.
assemble_code(get_integer(I, a(Ai)))     --> [[0x18,   Ai,   I1,   I0]], { sint(I, I1, I0) }.
assemble_code(get_integer(I, a(Ai)))     --> [[0x19,   Ai,    S,    N]], wsint(I, S, N).

assemble_code(set_variable(x(N)))     --> [[0x20, 0x00, 0x00,    N]].
assemble_code(set_variable(y(N)))     --> [[0x21, 0x00, 0x00,    N]].
assemble_code(set_value(x(N)))        --> [[0x22, 0x00, 0x00,    N]].
assemble_code(set_value(y(N)))        --> [[0x23, 0x00, 0x00,    N]].
assemble_code(set_constant(C))        --> [[0x27, 0x00,   C1,   C0]], { uint(C, C1, C0) }.
assemble_code(set_integer(I))         --> [[0x28, 0x00,   I1,   I0]], { sint(I, I1, I0) }.
assemble_code(set_integer(I))         --> [[0x29, 0x00,    S,    N]], wsint(I, S, N).
assemble_code(set_void(N))            --> [[0x2a, 0x00, 0x00,    N]].

assemble_code(unify_variable(x(N)))   --> [[0x30, 0x00, 0x00,    N]].
assemble_code(unify_variable(y(N)))   --> [[0x31, 0x00, 0x00,    N]].
assemble_code(unify_value(x(N)))      --> [[0x32, 0x00, 0x00,    N]].
assemble_code(unify_value(y(N)))      --> [[0x33, 0x00, 0x00,    N]].
assemble_code(unify_constant(C))      --> [[0x37, 0x00,   C1,   C0]], { uint(C, C1, C0) }.
assemble_code(unify_integer(I))       --> [[0x38, 0x00,   I1,   I0]], { sint(I, I1, I0) }.
assemble_code(unify_integer(I))       --> [[0x39, 0x00,    S,    N]], wsint(I, S, N).
assemble_code(unify_void(N))          --> [[0x3a, 0x00, 0x00,    N]].

assemble_code(allocate(N))        --> [[0x40, 0x00, 0x00,    N]].
assemble_code(trim(N))            --> [[0x41, 0x00, 0x00,    N]].
assemble_code(deallocate)         --> [[0x42, 0x00, 0x00, 0x00]].
assemble_code(label(L))           --> [label(L)].
assemble_code(call(ID, Arity))    --> [[0x43, Arity, program_location(ID)]].
assemble_code(execute(ID, Arity)) --> [[0x44, Arity, program_location(ID)]].
assemble_code(proceed)            --> [[0x45, 0x00, 0x00, 0x00]].

assemble_code(try_me_else(ID))   --> [[0x50, 0x00, program_location(ID)]].
assemble_code(retry_me_else(ID)) --> [[0x51, 0x00, program_location(ID)]].
assemble_code(trust_me)          --> [[0x52, 0x00, 0x00, 0x00]].
assemble_code(neck_cut)          --> [[0x53, 0x00, 0x00, 0x00]].
assemble_code(get_level(y(Yn)))  --> [[0x54, 0x00, 0x00,   Yn]].
assemble_code(cut(y(Yn)))        --> [[0x55, 0x00, 0x00,   Yn]].

assemble_code(>)   --> [[0x60, 0x00, 0x00, 0x00]].
assemble_code(<)   --> [[0x61, 0x00, 0x00, 0x00]].
assemble_code(=<)  --> [[0x62, 0x00, 0x00, 0x00]].
assemble_code(>=)  --> [[0x63, 0x00, 0x00, 0x00]].
assemble_code(=\=) --> [[0x64, 0x00, 0x00, 0x00]].
assemble_code(=:=) --> [[0x65, 0x00, 0x00, 0x00]].
assemble_code(is)  --> [[0x66, 0x00, 0x00, 0x00]].

assemble_code(true) --> [[0x70, 0x00, 0x00, 0x00]].
assemble_code(fail) --> [[0x71, 0x00, 0x00, 0x00]].
assemble_code(=)    --> [[0x72, 0x00, 0x00, 0x00]].

assemble_code(digital_input)          --> [[0x80, 0x00, 0x00, 0x00]].
assemble_code(digital_output)         --> [[0x80, 0x00, 0x00, 0x01]].
assemble_code(digital_input_pullup)   --> [[0x80, 0x00, 0x00, 0x02]].
assemble_code(digital_input_pulldown) --> [[0x80, 0x00, 0x00, 0x03]].
assemble_code(digital_read)           --> [[0x81, 0x00, 0x00, 0x00]].
assemble_code(digital_write)          --> [[0x82, 0x00, 0x00, 0x00]].


assemble_program_locations(Word_Components_With_Program_Locations, Word_Components) :-
	select([A, B, program_location(L)], Word_Components_With_Program_Locations, [A, B, C, D], Word_Components_With_Program_Locations1),
	!,
	uint(L, C, D),
	assemble_program_locations(Word_Components_With_Program_Locations1, Word_Components).

assemble_program_locations(Word_Components, Word_Components).


uint(I, I) :-
	I >= 0,
	I < 256.

uint(I, I1, I0) :-
	I >= 0,
	divmod(I, 256, I1, I0),
	I1 < 256.


uint(I, I3, I2, I1, I0) :-
	I >= 0,
	divmod(I, 65536, I32, I10),
	uint(I32, I3, I2),
	uint(I10, I1, I0).


sint(SI, I1, I0) :-
	SI >= -32768,
	SI < 32768,
	(   SI >= 0
	->  UI = SI
	;   UI = 65536 + SI),
	uint(UI, I1, I0).



wuint(0, Length, Length, Rest, Rest) :-
	!.
wuint(I, Length, Length_Acc, Is, Rest) :-
	divmod(I, 4294967296, IRest, ILast),
	uint(ILast, I3, I2, I1, I0),
	New_Acc is Length_Acc + 1,
	wuint(IRest, Length, New_Acc, Is, [[I3, I2, I1, I0]|Rest]).



wsint(SI, Sign, Length, Is, Rest) :-
	(   SI >= 0
	->  Sign = 0,
	    wuint(SI, Length, 0, Is, Rest)
	;   Sign = 1,
	    UI is -SI,
	    wuint(UI, Length, 0, Is, Rest)
	).


is_valid_word(W) :-
	is_list(W),
	length(W, 4),
	maplist(is_valid_word_component, W).


is_valid_word_component(C) :-
	integer(C),
	C >= 0,
	C < 256.


assembly_state(state(Functors, Labels), Functors, Labels).

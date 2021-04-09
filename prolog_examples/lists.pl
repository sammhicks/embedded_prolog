member(A, [A|_]).

member(A, [_|As]) :-
    member(A, As).


reverse(As, Bs) :-
    reverse(As, [], Bs).


reverse([], Bs, Bs) :- !.
reverse([A|As], Bs, BsFinal) :-
    reverse(As, [A|Bs], BsFinal).



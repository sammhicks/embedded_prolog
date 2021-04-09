outer_trim_test(AB, S) :-
    ab(AB),
    trim_test(S).

ab(a).
ab(b).


trim_test(S) :-
    abc(A, B, C),
    wrap(A, B, AB),
    abc(D, E, F),
    wrap(AB, D, ABD),
    wrap(ABD, E, ABDE),
    wrap(ABDE, C, ABDEC),
    wrap(ABDEC, F, S).

abc(a, b, c).

wrap(A, B, w(A, B)).


same(A, A).

foo(A, B, A, D, D) :-
    a(A),
    b(B).

a(a).
b(b).


bar(foo(A, x), A, A).

p(_, g(_), f(_, _, _)).

chained(l(A, B, f(C, A), D)) :-
    chainedA(A),
    chainedC(A, D),
    chainedD(D),
    chainedB(b(B, C)).

chainedA(a(x, _)).
chainedB(b(C, c(C))).
chainedC(X, X).
chainedD(a(_, y)).


also_chained(A, C, E) :-
    also_chainedA(A, a(b(C), d(e)), a(b(c), d(E))).

also_chainedA(A, A, A).

deep_chain(A, a(b(c(d(e(f(g(A))))))), a(b(c(d(e(f(g(A)))))))).


other_test(X) :-
    also_chainedA(X, a(b(c(d(e(a, _))))), a(b(c(d(e(_, b)))))).

abc(a(A)) :- aaba(A).
abc(b(a)) :- !.
abc(b(b)).
abc(c(a)).
abc(c(b)).

aaba(a).
aaba(b).


cut_test(A) :-
    aaba(A),
    !,
    aaba(A).


cut_test(A, B) :-
    cut_test_a(A),
    !,
    cut_test_b(A, B).


cut_test_a(a).
cut_test_a(b).
cut_test_a(c).

cut_test_b(A, b(a(A))).
cut_test_b(A, b(b(A))).


windback_test(A) :-
    windback_test_a(A),
    windback_test_b(A).

windback_test_a(a(b(c), b(c))) :- fail.
windback_test_a(a(b(c), b(c))).

windback_test_b(a(b(c), b(c))).

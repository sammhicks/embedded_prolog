on_line(bond_street, central).
on_line(bond_street, jubilee).
on_line(charing_cross, bakerloo).
on_line(charing_cross, jubilee).
on_line(charing_cross, northern).
on_line(green_park, jubilee).
on_line(green_park, piccadilly).
on_line(green_park, victoria).
on_line(leicester_square, northern).
on_line(leicester_square, piccadilly).
on_line(oxford_circus, bakerloo).
on_line(oxford_circus, central).
on_line(oxford_circus, victoria).
on_line(piccadilly_circus, bakerloo).
on_line(piccadilly_circus, piccadilly).
on_line(tottenham_court_road, central).
on_line(tottenham_court_road, northern).

same_line(Station1, Station2, Line) :-
	on_line(Station1, Line),
	on_line(Station2, Line).


journey_to(From, To, Route) :-
	journey_to(From, To, Route, [station(From)]).


journey_to(From, To, [(Line:(From-To))], Already_Visited) :-
	same_line(From, To, Line),
	not_member(Already_Visited, line(Line)),
	not_member(Already_Visited, station(To)).

journey_to(From, To, [(Line:(From-Intermediate))|Route], Already_Visited) :-
	same_line(From, Intermediate, Line),
	not_member(Already_Visited, station(Intermediate)),
	not_member(Already_Visited, line(Line)),
	journey_to(Intermediate, To, Route, [station(Intermediate),line(Line)|Already_Visited]).


not_member([], _Item) :-
	!.

not_member([Item|_], Item) :-
	!,
	fail.

not_member([_|Items], Item) :-
	not_member(Items, Item).

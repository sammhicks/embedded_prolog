use std::{collections::HashMap, fmt, hash::Hash};

use arcstr::ArcStr;
use num_bigint::BigInt;

struct CountingSet<T>(HashMap<T, usize>);

impl<T: Eq + Hash> CountingSet<T> {
    fn new() -> Self {
        Self(HashMap::new())
    }

    fn insert(&mut self, value: T) {
        self.0
            .entry(value)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }

    fn get_with_count_one(&self) -> impl Iterator<Item = &T> {
        self.0
            .iter()
            .filter_map(move |(value, count)| (*count == 1).then_some(value))
    }
}

impl<T: Eq + Hash> Extend<T> for CountingSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.insert(value)
        }
    }
}

pub const EMPTY_LIST: ArcStr = arcstr::literal!("[|]");

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VariableName<'a> {
    Named(&'a str),
    Cut,
}

#[derive(Debug)]
pub enum Term {
    Variable { name: ArcStr },
    Structure { name: ArcStr, terms: TermList },
    List { head: Box<Term>, tail: Box<Term> },
    Constant { name: ArcStr },
    Integer { i: BigInt },
    Void,
}

impl Term {
    pub const EMPTY_LIST: Term = Term::Constant { name: EMPTY_LIST };

    pub fn list(head: Term, tail: Term) -> Term {
        Term::List {
            head: Box::new(head),
            tail: Box::new(tail),
        }
    }

    pub fn variables<'a, V: Extend<VariableName<'a>>>(&'a self, mut variables: V) -> V {
        match self {
            Self::Variable { name } => {
                variables.extend(std::iter::once(VariableName::Named(name.as_str())));
                variables
            }
            Self::Structure { terms, .. } => terms.variables(variables),
            Self::List { head, tail } => head.variables(tail.variables(variables)),
            Self::Constant { .. } | Self::Integer { .. } | Self::Void => variables,
        }
    }

    pub fn make_variables_void(&mut self, variables: &[String]) {
        match self {
            Term::Variable { name } => {
                if variables.iter().any(|variable| variable.as_str() == name) {
                    println!("{name} is now void");
                    *self = Term::Void
                }
            }
            Term::Structure { terms, .. } => {
                terms.make_variables_void(variables);
            }
            Term::List { head, tail } => {
                head.make_variables_void(variables);
                tail.make_variables_void(variables)
            }
            Term::Constant { .. } | Term::Integer { .. } | Term::Void => (),
        }
    }
}

#[derive(Default)]
pub struct TermList(Vec<Term>);

impl fmt::Debug for TermList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::ops::Deref for TermList {
    type Target = [Term];

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl From<Vec<Term>> for TermList {
    fn from(terms: Vec<Term>) -> Self {
        Self(terms)
    }
}

impl TermList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn variables<'a, V: Extend<VariableName<'a>>>(&'a self, variables: V) -> V {
        self.0
            .iter()
            .fold(variables, |variables, term| term.variables(variables))
    }

    pub fn make_variables_void(&mut self, variables: &[String]) {
        for term in self.0.iter_mut() {
            term.make_variables_void(variables);
        }
    }
}

impl<'a> IntoIterator for &'a TermList {
    type Item = &'a Term;
    type IntoIter = std::slice::Iter<'a, Term>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CallName<Name> {
    Named(Name),
    True,
    Fail,
    Is,
}

impl<'a> CallName<&'a str> {
    pub fn map_name<O: 'a, F: FnOnce(&'a str) -> O>(self, f: F) -> CallName<O> {
        match self {
            CallName::Named(name) => CallName::Named(f(name)),
            CallName::True => CallName::True,
            CallName::Fail => CallName::Fail,
            CallName::Is => CallName::Is,
        }
    }
}

impl<Name: AsRef<str>> CallName<Name> {
    pub fn as_ref(&self) -> CallName<&str> {
        match self {
            CallName::Named(name) => CallName::Named(name.as_ref()),
            CallName::True => CallName::True,
            CallName::Fail => CallName::Fail,
            CallName::Is => CallName::Is,
        }
    }
}

impl<T: fmt::Display> fmt::Display for CallName<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Named(name) => name.fmt(f),
            Self::True => write!(f, "true"),
            Self::Fail => write!(f, "fail"),
            Self::Is => write!(f, "'is'"),
        }
    }
}

#[derive(Debug)]
pub enum Goal<Name: AsRef<str>> {
    Named {
        name: CallName<Name>,
        terms: TermList,
    },
    NeckCut,
    Cut,
}

impl<Name: AsRef<str>> Goal<Name> {
    pub fn variables<'a, V: Extend<VariableName<'a>>>(&'a self, mut variables: V) -> V {
        match self {
            Goal::Named { terms, .. } => terms.variables(variables),
            Goal::NeckCut => variables,
            Goal::Cut => {
                variables.extend(std::iter::once(VariableName::Cut));
                variables
            }
        }
    }

    pub fn make_variables_void(&mut self, variables: &[String]) {
        match self {
            Goal::Named { terms, .. } => terms.make_variables_void(variables),
            Goal::NeckCut | Goal::Cut => (),
        }
    }
}

#[derive(Default)]
pub struct GoalList(Vec<Goal<ArcStr>>);

impl fmt::Debug for GoalList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::ops::Deref for GoalList {
    type Target = [Goal<ArcStr>];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl From<Vec<Goal<ArcStr>>> for GoalList {
    fn from(mut goals: Vec<Goal<ArcStr>>) -> Self {
        if let Some(first @ Goal::Cut) = goals.first_mut() {
            *first = Goal::NeckCut
        }

        Self(goals)
    }
}

impl IntoIterator for GoalList {
    type Item = Goal<ArcStr>;
    type IntoIter = <Vec<Goal<ArcStr>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl GoalList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn variables<'a, V: Extend<VariableName<'a>>>(&'a self, variables: V) -> V {
        self.0
            .iter()
            .fold(variables, |variables, goal| goal.variables(variables))
    }

    fn make_variables_void(&mut self, variables: &[String]) {
        for goal in self.0.iter_mut() {
            goal.make_variables_void(variables);
        }
    }
}

#[derive(Debug)]
pub struct Definition {
    pub head: TermList,
    pub body: GoalList,
}

impl Definition {
    pub fn with_singletons_removed(mut self) -> Self {
        let variables = self.head.variables(self.body.variables(CountingSet::new()));

        let variables = variables
            .get_with_count_one()
            .filter_map(|name| match *name {
                VariableName::Named(name) => Some(String::from(name)),
                VariableName::Cut => None,
            })
            .collect::<Vec<_>>();

        self.head.make_variables_void(&variables);
        self.body.make_variables_void(&variables);

        self
    }
}

#[derive(Debug)]
pub enum Disjunction {
    Single {
        name: ArcStr,
        arity: u8,
        definition: Definition,
    },
    Multiple {
        name: ArcStr,
        arity: u8,
        first_definition: Definition,
        middle_definitions: Vec<Definition>,
        last_definition: Definition,
    },
}

#[derive(Debug)]
pub struct Program {
    pub definitions: Vec<Disjunction>,
}

#[derive(Debug)]
pub struct Query {
    pub name: CallName<ArcStr>,
    pub terms: TermList,
}

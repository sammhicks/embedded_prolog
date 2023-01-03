use std::{borrow::Borrow, collections::HashMap, fmt, hash::Hash, ops::Range, path::Path, rc::Rc};

use arcstr::ArcStr;
use num_bigint::BigInt;

use super::SortExt;

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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceId {
    SystemCall,
    Program(Rc<Path>),
    Query,
}

impl fmt::Display for SourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SourceId::SystemCall => write!(f, "<system call>"),
            SourceId::Program(path) => write!(f, "{}", path.display()),
            SourceId::Query => write!(f, "<query>"),
        }
    }
}

impl SourceId {
    #[allow(clippy::type_complexity)]
    pub fn stream<'a>(
        self,
        source: &'a str,
    ) -> chumsky::Stream<
        'a,
        char,
        (Self, Range<usize>),
        Box<dyn Iterator<Item = (char, (Self, Range<usize>))> + 'a>,
    > {
        let len = source.len();

        chumsky::Stream::from_iter(
            (self.clone(), len..len),
            Box::new(
                source
                    .chars()
                    .enumerate()
                    .map(move |(i, c)| (c, (self.clone(), i..i + 1))),
            ),
        )
    }
}

#[derive(Clone)]
pub struct Name {
    value: ArcStr,
    span: (SourceId, Range<usize>),
}

impl Name {
    pub fn new<S: Borrow<str>>(value: S, span: (SourceId, Range<usize>)) -> Self {
        Self {
            value: value.borrow().into(),
            span,
        }
    }

    pub fn as_string(&self) -> ArcStr {
        self.value.clone()
    }

    pub fn span(&self) -> &(SourceId, Range<usize>) {
        &self.span
    }
}

impl From<Name> for ArcStr {
    fn from(name: Name) -> Self {
        name.value
    }
}

impl<'a> From<&'a Name> for ArcStr {
    fn from(name: &'a Name) -> Self {
        name.value.clone()
    }
}

impl AsRef<str> for Name {
    fn as_ref(&self) -> &str {
        &self.value
    }
}

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.value.fmt(f)
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.value.fmt(f)
    }
}

impl PartialEq for Name {
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
}

impl Eq for Name {}

impl PartialOrd for Name {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Name {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl Hash for Name {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state)
    }
}

pub const EMPTY_LIST: ArcStr = arcstr::literal!("[]");

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VariableName<'a> {
    Named(&'a Name),
    Cut,
}

#[derive(Debug)]
pub enum Term {
    Variable { name: Name },
    Structure { name: Name, terms: TermList },
    List { head: Box<Term>, tail: Box<Term> },
    Constant { name: Name },
    Integer { i: BigInt },
    Void,
}

impl Term {
    pub fn list(head: Term, tail: Term) -> Term {
        Term::List {
            head: Box::new(head),
            tail: Box::new(tail),
        }
    }

    pub fn variables<'a, V: Extend<VariableName<'a>>>(&'a self, mut variables: V) -> V {
        match self {
            Self::Variable { name } => {
                variables.extend(std::iter::once(VariableName::Named(name)));
                variables
            }
            Self::Structure { terms, .. } => terms.variables(variables),
            Self::List { head, tail } => head.variables(tail.variables(variables)),
            Self::Constant { .. } | Self::Integer { .. } | Self::Void => variables,
        }
    }

    pub fn make_variables_void(&mut self, variables: &[Name]) {
        match self {
            Term::Variable { name } => {
                if variables.iter().any(|variable| variable == name) {
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

impl<const N: usize> From<[Term; N]> for TermList {
    fn from(terms: [Term; N]) -> Self {
        Self(terms.into())
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

    pub fn make_variables_void(&mut self, variables: &[Name]) {
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
    Unify,
    Is,
    Comparison(super::instructions::Comparison),
}

impl<'a> CallName<&'a Name> {
    pub fn map_name<O: 'a, F: FnOnce(&'a Name) -> O>(self, f: F) -> CallName<O> {
        match self {
            Self::Named(name) => CallName::Named(f(name)),
            Self::True => CallName::True,
            Self::Fail => CallName::Fail,
            Self::Unify => CallName::Unify,
            Self::Is => CallName::Is,
            Self::Comparison(comparison) => CallName::Comparison(comparison),
        }
    }
}

impl CallName<Name> {
    pub fn as_ref(&self) -> CallName<&Name> {
        match *self {
            Self::Named(ref name) => CallName::Named(name),
            Self::True => CallName::True,
            Self::Fail => CallName::Fail,
            Self::Unify => CallName::Unify,
            Self::Is => CallName::Is,
            Self::Comparison(comparison) => CallName::Comparison(comparison),
        }
    }
}

impl<T: fmt::Display> fmt::Display for CallName<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Named(name) => name.fmt(f),
            Self::True => "true".fmt(f),
            Self::Fail => "fail".fmt(f),
            Self::Unify => "=".fmt(f),
            Self::Is => "is".fmt(f),
            Self::Comparison(comparison) => comparison.fmt(f),
        }
    }
}

#[derive(Debug)]
pub enum Goal<N: AsRef<str>> {
    Named { name: CallName<N>, terms: TermList },
    NeckCut,
    Cut,
}

impl<N: AsRef<str>> Goal<N> {
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

    pub fn make_variables_void(&mut self, variables: &[Name]) {
        match self {
            Goal::Named { terms, .. } => terms.make_variables_void(variables),
            Goal::NeckCut | Goal::Cut => (),
        }
    }
}

#[derive(Default)]
pub struct GoalList(Vec<Goal<Name>>);

impl fmt::Debug for GoalList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::ops::Deref for GoalList {
    type Target = [Goal<Name>];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl From<Vec<Goal<Name>>> for GoalList {
    fn from(mut goals: Vec<Goal<Name>>) -> Self {
        if let Some(first @ Goal::Cut) = goals.first_mut() {
            *first = Goal::NeckCut
        }

        Self(goals)
    }
}

impl GoalList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn variables<'a, V: Extend<VariableName<'a>>>(&'a self, variables: V) -> V {
        self.0
            .iter()
            .fold(variables, |variables, goal| goal.variables(variables))
    }

    fn make_variables_void(&mut self, variables: &[Name]) {
        for goal in self.0.iter_mut() {
            goal.make_variables_void(variables);
        }
    }
}

#[derive(Debug)]
pub struct Clause {
    pub head: TermList,
    pub body: GoalList,
}

impl Clause {
    pub fn with_singletons_removed(mut self, source_id: SourceId, source: &str) -> Self {
        use chumsky::Span;

        let variables = self.head.variables(self.body.variables(CountingSet::new()));

        let variables = variables
            .get_with_count_one()
            .filter_map(|name| match *name {
                VariableName::Named(name) => Some(name.clone()),
                VariableName::Cut => None,
            })
            .collect::<Vec<_>>()
            .sorted_by_key(|variable| variable.span().start());

        if !variables.is_empty() {
            let mut colors = ariadne::ColorGenerator::new();

            for variable in &variables {
                let msg = format!("{} is a Singleton", variable.as_ref());

                ariadne::Report::build(
                    ariadne::ReportKind::Warning,
                    source_id.clone(),
                    variable.span().start(),
                )
                .with_message(&msg)
                .with_label(
                    ariadne::Label::new(variable.span().clone())
                        .with_color(colors.next())
                        .with_message(msg),
                )
                .finish()
                .print((source_id.clone(), ariadne::Source::from(source)))
                .unwrap();
            }
        }

        self.head.make_variables_void(&variables);
        self.body.make_variables_void(&variables);

        self
    }
}

#[derive(Debug)]
pub struct Definition {
    pub name: Name,
    pub arity: u8,
    pub clauses: Vec<Clause>,
}

#[derive(Debug)]
pub struct Program {
    pub definitions: Vec<Definition>,
}

#[derive(Debug)]
pub struct Query {
    pub name: CallName<Name>,
    pub terms: TermList,
}

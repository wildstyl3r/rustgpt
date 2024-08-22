use std::collections::{BTreeMap, BTreeSet};
pub struct Vocabulary {
    pub stoi: BTreeMap<String, i64>,
    pub itos: Vec<String>
}

impl Vocabulary {
    pub fn new(s: &str) -> Self {
        let mut set = BTreeSet::new();
        for c in s.chars() {
            set.insert(c);
        }
        let mut stoi = BTreeMap::new();
        let mut itos = Vec::new();
        for (i, c) in set.iter().enumerate() {
            itos.push(c.to_string());
            stoi.insert(c.to_string(), i as i64);
        }
        Vocabulary { stoi, itos}
    }

    pub fn len(&self) -> usize {
        self.stoi.len()
    }
}
use std::collections::{HashMap, HashSet};

pub type Token = i64;

pub struct Tokenizer {
    pub vocabulary: Vec<char>,
    pub char_to_token: HashMap<char, Token>,
    pub token_to_char: HashMap<Token, char>,
}

impl Tokenizer {
    pub fn new(s: &str) -> Self {
        let mut vocabulary: Vec<char> = s.chars().collect::<HashSet<char>>().into_iter().collect();
        vocabulary.sort();
        Tokenizer {
            char_to_token: vocabulary
                .iter()
                .enumerate()
                .map(|(i, &c)| (c, i as Token))
                .collect(),
            token_to_char: vocabulary
                .iter()
                .enumerate()
                .map(|(i, &c)| (i as Token, c))
                .collect(),
            vocabulary,
        }
    }

    pub fn encode(&self, s: &str) -> Vec<Token> {
        s.chars()
            .map(|c| {
                *self
                    .char_to_token
                    .get(&c)
                    .unwrap_or(self.char_to_token.get(&' ').unwrap())
            })
            .collect()
    }

    pub fn decode(&self, v: &[Token]) -> String {
        v.iter()
            .map(|c| {
                *self
                    .token_to_char
                    .get(c)
                    .unwrap_or(self.token_to_char.get(&0).unwrap())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_both_ways() {
        let mock = "hello world";
        let tokenizer = Tokenizer::new(mock);
        assert_eq!(mock, tokenizer.decode(&tokenizer.encode(mock)));
    }
}

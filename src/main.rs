use std::collections::HashMap;

fn normalize(s: String) -> String {
    todo!();
}

fn encode(s: String, vocabulary: Option<HashMap<String, u32>>) -> Vec<u32> {
    match vocabulary {
        Some(v) => todo!(),
        None =>s.chars().map(|c| c as u32).collect()
    }
}

fn decode(v: Vec<u32>, vocabulary: Option<HashMap<String, u32>>) -> String {
    match vocabulary {
        Some(v) => todo!(),
        None => v.iter().filter_map(|i| char::from_u32(*i)).collect()
    }
}

fn main() {
    println!("Hello, world!");
}

use funclog::funclog;
use tomlstruct::tomlstruct;
use typename::{TypeName, TypeNameTrait};

#[derive(TypeName)]
struct Hello;

#[funclog]
fn hello() {
    println!("Hello, world!");
}

fn main() {
    let x = Hello;
    dbg!(x.type_name());

    hello();
}

let greet name = Printf.sprintf "hello, %s, from inside the devcontainer" name
let () = print_endline (greet "world")

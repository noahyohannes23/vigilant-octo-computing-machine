open Micrograd

(* A small example graph: a single neuron, tanh (x1*w1 + x2*w2 + b). *)
let build_example () =
  let x1 = input_data 2.0 in
  let x2 = input_data 0.0 in
  let w1 = input_data (-3.0) in
  let w2 = input_data 1.0 in
  let b = input_data 6.8813735870195432 in
  let n = (x1 * w1) + (x2 * w2) + b in
  tanh n
;;

(* Write [dot] to [dot_file] and, if the `dot` binary is available, render it to
   an SVG next to it. *)
let render dot_file svg_file dot =
  let oc = open_out dot_file in
  output_string oc dot;
  close_out oc;
  Printf.printf "Wrote %s\n" dot_file;
  let cmd =
    Printf.sprintf
      "dot -Tsvg %s -o %s"
      (Filename.quote dot_file)
      (Filename.quote svg_file)
  in
  match Sys.command cmd with
  | 0 -> Printf.printf "Wrote %s\n" svg_file
  | code ->
    Printf.printf
      "`dot` exited with %d (is graphviz installed? `apt-get install \
       graphviz`).\n\
       The DOT source is still available at %s.\n"
      code
      dot_file
;;

let () =
  let out = build_example () in
  set_grad out 1.0;
  backward_pass out;
  render "graph.dot" "graph.svg" (to_dot out)
;;

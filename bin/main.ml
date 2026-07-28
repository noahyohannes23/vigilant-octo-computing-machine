open Micrograd

(* Karpathy's toy binary-classification dataset: 4 rows of 3 features, targets
   in {-1, +1}. *)
let xs =
  [ [ 2.0; 3.0; -1.0 ]
  ; [ 3.0; -1.0; 0.5 ]
  ; [ 0.5; 1.0; 1.0 ]
  ; [ 1.0; 1.0; -1.0 ]
  ]
;;

let ys = [ 1.0; -1.0; -1.0; 1.0 ]

(* A small, human-readable graph: a single neuron tanh(x1*w1 + x2*w2 + b). *)
let single_neuron_graph () =
  let x1 = input 2.0
  and x2 = input 0.0
  and w1 = input (-3.0)
  and w2 = input 1.0
  and b = input 6.8813735870195432 in
  tanh ((x1 *& w1) +& (x2 *& w2) +& b)
;;

(* Write [dot] to [dot_file] and, if the `dot` binary is available, render it to
   an SVG next to it. *)
let render dot_file svg_file dot =
  let oc = open_out dot_file in
  output_string oc dot;
  close_out oc;
  Printf.printf "Wrote %s\n" dot_file;
  let cmd =
    Printf.sprintf "dot -Tsvg %s -o %s" (Filename.quote dot_file) (Filename.quote svg_file)
  in
  match Sys.command cmd with
  | 0 -> Printf.printf "Wrote %s\n" svg_file
  | code ->
    Printf.printf
      "`dot` exited with %d (is graphviz installed? `apt-get install graphviz`).\n\
       The DOT source is still available at %s.\n"
      code
      dot_file
;;

let () =
  Random.self_init ();
  (* 1. Train a small MLP (3 -> 4 -> 4 -> 1) on the toy dataset. *)
  let net = create_mlp 3 [ 4; 4; 1 ] in
  let epochs = 100 in
  let trained, history = train ~lr:0.1 ~epochs xs ys net in
  List.iteri
    (fun i l ->
       if i = 0 || (i + 1) mod 20 = 0
       then Printf.printf "epoch %3d  loss %.6f\n" (i + 1) l)
    history;
  Printf.printf "\npredictions vs targets:\n";
  List.iter2
    (fun xrow y ->
       Printf.printf "  pred % .4f   target % .1f\n" (predict trained xrow).data y)
    xs
    ys;
  (* 2. Render a small computation graph annotated with its gradients. *)
  Printf.printf "\n";
  let g = single_neuron_graph () in
  render "graph.dot" "graph.svg" (to_dot ~grads:(backward g) g)
;;

(* Micrograd implementation in OCaml *)

(* data types *)

type op = string
type grad = float ref
type data = float
type backward = unit -> unit
type value0 = Node of data * grad * backward * op * value0 list

(* base datatypes *)

let input_data data = Node (data, ref 0., (fun () -> ()), "", [])

(* operations on values *)

let ( +& ) (Node (d1, g1, _, _, _) as v1) (Node (d2, g2, _, _, _) as v2) =
  let grad = ref 0. in
  let backward () =
    g1 := !g1 +. !grad;
    g2 := !g2 +. !grad;
    ()
  in
  Node (d1 +. d2, grad, backward, "+", [ v1; v2 ])
;;

let ( *& ) (Node (d1, g1, _, _, _) as v1) (Node (d2, g2, _, _, _) as v2) =
  let grad = ref 0. in
  let backward () =
    g1 := !g1 +. (d2 *. !grad);
    g2 := !g2 +. (d1 *. !grad);
    ()
  in
  Node (d1 *. d2, grad, backward, "*", [ v1; v2 ])
;;

let tanh (Node (d, g1, _, _, _) as v) =
  let t = (exp (2. *. d) -. 1.) /. (exp (2. *. d) +. 1.) in
  let grad = ref 0. in
  let backward () =
    g1 := !g1 +. ((1. -. (t ** 2.)) *. !grad);
    ()
  in
  Node (t, grad, backward, "tanh", [ v ])
;;

let neg v = v *& input_data (-1.)
let ( -& ) v1 v2 = v1 +& neg v2

let ( **& ) (Node (d, g, _, _, _) as v) (x : float) =
  let grad = ref 0. in
  let backward () =
    g := !g +. (x *. (d ** (x -. 1.0)) *. !grad);
    ()
  in
  Node (d ** x, grad, backward, "**", [ v ])
;;

(* backward pass *)

(* Run backprop over the whole DAG rooted at [root].

   [build] produces a reverse-topological order (each node before its children):
   for a DAG, membership in the accumulator doubles as the visited set -- a node
   lands in it exactly once, and there is no path back to a node to re-enter it.
   Prepending after folding the children puts every parent ahead of its
   children, so no final reverse is needed.

   Gradients accumulate (the closures use +=), so we must zero every node first;
   then we seed the output's grad to 1 and fire each node's [backward] once, in
   order, guaranteeing a node's grad is final before it feeds its children. *)
let backward_pass root =
  let rec build order (Node (_, _, _, _, ch) as n) =
    if List.memq n order then order else n :: List.fold_left build order ch
  in
  let order = build [] root in
  List.iter (fun (Node (_, g, _, _, _)) -> g := 0.) order;
  (match root with
   | Node (_, g, _, _, _) -> g := 1.);
  List.iter (fun (Node (_, _, b, _, _)) -> b ()) order
;;

(* Seed the gradient of a node (typically the output, set to 1. before a
   backward pass). *)

let set_grad (Node (_, g, _, _, _)) v = g := v

(* Emit a Graphviz DOT description of the computation graph rooted at [root].

   The structure mirrors Karpathy's micrograd: every [Value] is drawn as a
   record box showing its data and grad, and every operation (+, *, tanh) is a
   small oval feeding its result box. Edges run from operand boxes into the op.

   Nodes are deduplicated by *physical* identity (== / List.memq), not
   structural equality: this is a DAG, so a single Value can feed several ops,
   and structural comparison would both mis-merge distinct nodes and loop
   forever chasing the [backward] closure. [ids] hands out a stable integer per
   physical node (assigned lazily, so an edge can reference a child before it is
   drawn), while [emitted] tracks which boxes have already been written. *)
let to_dot root =
  let buf = Buffer.create 256 in
  let ids : (value0 * int) list ref = ref [] in
  let next = ref 0 in
  let id_of n =
    match List.assq_opt n !ids with
    | Some i -> i
    | None ->
      let i = !next in
      incr next;
      ids := (n, i) :: !ids;
      i
  in
  let emitted : value0 list ref = ref [] in
  let rec go (Node (d, g, _, op, children) as n) =
    if not (List.memq n !emitted)
    then (
      emitted := n :: !emitted;
      let id = id_of n in
      Printf.bprintf buf "  n%d [shape=record label=\"data %.4f | grad %.4f\"];\n" id d !g;
      if op <> ""
      then (
        Printf.bprintf buf "  op%d [label=%S];\n" id op;
        Printf.bprintf buf "  op%d -> n%d;\n" id id);
      List.iter
        (fun c ->
           let cid = id_of c in
           if op = ""
           then Printf.bprintf buf "  n%d -> n%d;\n" cid id
           else Printf.bprintf buf "  n%d -> op%d;\n" cid id)
        children;
      List.iter go children)
  in
  Buffer.add_string buf "digraph G {\n  rankdir=LR;\n";
  go root;
  Buffer.add_string buf "}\n";
  Buffer.contents buf
;;

type weights = value0 list
type neuron = weights * value0

(* initialize random module for create neuron function *)

let () = Random.self_init ()

(* neuron functions *)

let create_neuron (n_in : int) : neuron =
  let uv _ =
    input_data
    @@ Float.mul (Random.float 1.)
    @@ float_of_int
    @@ if Random.int 1 = 0 then -1 else 1
  in
  let ws = List.of_seq @@ Seq.init n_in uv in
  ws, uv ()
;;

let call_neuron (inputs : value0 list) ((weights, bias) : neuron) : value0 =
  List.fold_left
    (fun sum (w_i, x_i) -> sum +& (w_i *& x_i))
    bias
    (List.combine weights inputs)
;;

(* layer functions *)

type layer = neuron list

let create_layer (n_in : int) (n_out : int) : layer =
  List.of_seq @@ Seq.init n_out (fun _ -> create_neuron n_in)
;;

let call_layer (inputs : value0 list) = fun (l : layer) -> List.map (call_neuron inputs) l

(* helper function that helps iterate on two items instead of one *)

let rec map_2el f l =
  match l with
  | [] -> []
  | [ _ ] -> []
  | fst :: snd :: tl -> f fst snd :: map_2el f (snd :: tl)
;;

(* mlp functions *)

type mlp = layer list

let create_mlp (n_in : int) (n_outs : int list) : mlp =
  map_2el create_layer (n_in :: n_outs)
;;

let call_mlp (inputs : value0 list) = fun (m : mlp) -> List.fold_left call_layer inputs m

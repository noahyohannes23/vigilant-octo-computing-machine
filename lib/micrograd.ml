type op = string
type grad = float ref
type data = float
type backward = unit -> unit
type value0 = Node of data * grad * backward * op * value0 list

let input_data data = Node (data, ref 0., (fun () -> ()), "", [])

let ( +& ) (Node (d1, g1, _, _, _) as v1) (Node (d2, g2, _, _, _) as v2) =
  let grad = ref 0. in
  let backward () =
    g1 := 1. *. !grad;
    g2 := 1. *. !grad;
    ()
  in
  Node (d1 +. d2, grad, backward, "+", [ v1; v2 ])
;;

let ( *& ) (Node (d1, g1, _, _, _) as v1) (Node (d2, g2, _, _, _) as v2) =
  let grad = ref 0. in
  let backward () =
    g1 := d2 *. !grad;
    g2 := d1 *. !grad;
    ()
  in
  Node (d1 *. d2, grad, backward, "*", [ v1; v2 ])
;;

let tanh (Node (d, g1, _, _, _) as v) =
  let t = (exp (2. *. d) -. 1.) /. (exp (2. *. d) +. 1.) in
  let grad = ref 0. in
  let backward () =
    g1 := (1. -. (t ** 2.)) *. !grad;
    ()
  in
  Node (t, grad, backward, "tanh", [ v ])
;;

let neg v = v *& input_data (-1.)
let ( - ) v1 v2 = v1 +& neg v2

let ( ** ) (Node (d, g, _, _, _) as v) (x : float) =
  let grad = ref 0. in
  let backward () =
    g := ((x *. d) ** (x -. 1.0)) *. !grad;
    ()
  in
  Node (d ** x, grad, backward, "**", [ v ])
;;

let rec backward_pass (Node (_, _, b, _, c)) =
  b ();
  List.iter backward_pass c
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

let create_neuron (n_in : int) : neuron =
  Random.self_init ();
  let float_in_range r =
    let multiplier = if Random.int 1 = 0 then -1 else 1 in
    float_of_int multiplier *. Random.float r
  in
  let rec range f x =
    match f x with
    | None -> []
    | Some (res, next) -> res :: range f next
  in
  let uniform_value x = x |> float_in_range |> input_data in
  let irange (i : int) = if i > n_in then None else Some (uniform_value 1., i + 1) in
  range irange 1, uniform_value 1.
;;

type 'a gen_iter_opt = (int * 'a) option

(*
  bind ((i , x) : (int * 'a)) (f : 'a -> int * 'b) : int * 'b

  value: (int * 'a) -> ('a -> (int * 'b)) -> int * 'b

  think of the api

  takes a function (that takes a number i as an arugment and returns some value) and returns a generated list of that function applied to i=1 to some max value
  *)

let call_neuron ((weights, bias) : neuron) (inputs : value0 list) : value0 =
  List.fold_left
    (fun sum (w_i, x_i) -> sum +& (w_i *& x_i))
    bias
    (List.combine weights inputs)
;;

type layer = neuron list

let create_layer n_in n_out =
  let rec range f x =
    match f x with
    | None -> []
    | Some (res, next) -> res :: range f next
  in
  let irange i = if i > n_out then None else Some (create_neuron n_in, i + 1) in
  range irange 1
;;

let call_layer inputs = List.map (fun n -> call_neuron n inputs)

(* Figure out how to turn irange and range into more callable/normalized functions via monads / currying*)

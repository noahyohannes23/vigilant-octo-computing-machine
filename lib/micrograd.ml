(* Micrograd in OCaml -- a small reverse-mode autodiff engine plus a tiny neural
   net library, following Karpathy's micrograd.

   The engine is purely functional. A [value] is an immutable node in a
   computation DAG: its data, the op that produced it, and its operands. There
   are no ref cells and nothing is ever mutated -- so a [value] always denotes a
   fixed number, and backprop can be run any number of times with no need to
   "zero" anything first. The backward pass is a fold that *returns* a mapping
   from node to gradient rather than writing gradients back into the graph. *)

(* ----- values ----- *)

type op =
  | Input
  | Add
  | Mul
  | Tanh
  | Pow of float

type value =
  { data : float
  ; op : op
  ; prev : value list
  }

let input data = { data; op = Input; prev = [] }
let ( +& ) a b = { data = a.data +. b.data; op = Add; prev = [ a; b ] }
let ( *& ) a b = { data = a.data *. b.data; op = Mul; prev = [ a; b ] }

let tanh v =
  let t = (exp (2. *. v.data) -. 1.) /. (exp (2. *. v.data) +. 1.) in
  { data = t; op = Tanh; prev = [ v ] }
;;

let ( **& ) v (x : float) = { data = v.data ** x; op = Pow x; prev = [ v ] }
let neg v = v *& input (-1.)
let ( -& ) a b = a +& neg b

(* ----- backward pass ----- *)

(* Reverse-topological order: every node appears before its operands.

   Nodes are deduplicated by *physical* identity ([memq]), not structural
   equality: this is a shared DAG (one value can feed several ops), so structural
   comparison would both mis-merge distinct-but-equal nodes and diverge chasing
   cycles that don't exist in identity terms. Prepending [n] after folding its
   operands puts every parent ahead of its children, which is exactly the order
   backprop needs. *)
let topo root =
  let rec build order n =
    if List.memq n order then order else n :: List.fold_left build order n.prev
  in
  build [] root
;;

(* A gradient environment: an association list keyed by physical node identity.
   [add_grad] prepends, so the head binding for a node always holds its running
   sum and [assq] returns that head. This is what replaces the old grad refs --
   accumulation lives in a returned value, not in the graph. *)
type grads = (value * float) list

let grad_of (env : grads) node = Option.value ~default:0. (List.assq_opt node env)
let add_grad (env : grads) node delta : grads = (node, grad_of env node +. delta) :: env

(* Local partials d(output)/d(operand) for a single node, given the node's own
   value. On a [Tanh] node [data] is already tanh(x), so d/dx = 1 - data^2. *)
let local_grads node =
  match node.op, node.prev with
  | Input, _ -> []
  | Add, [ a; b ] -> [ a, 1.; b, 1. ]
  | Mul, [ a; b ] -> [ a, b.data; b, a.data ]
  | Tanh, [ x ] -> [ x, 1. -. (node.data ** 2.) ]
  | Pow e, [ x ] -> [ x, e *. (x.data ** (e -. 1.)) ]
  | _ -> invalid_arg "local_grads: malformed node"
;;

(* Backprop from [root]: seed its gradient to 1, then walk parents-before-
   children, pushing each node's accumulated gradient onto its operands via the
   chain rule. Because every parent precedes its children in [topo], a node's
   gradient is final by the time we read it. Returns the full gradient
   environment; query it with [grad_of]. Nothing in [root] is modified. *)
let backward root : grads =
  List.fold_left
    (fun env node ->
       let g = grad_of env node in
       List.fold_left
         (fun env (child, local) -> add_grad env child (local *. g))
         env
         (local_grads node))
    (add_grad [] root 1.)
    (topo root)
;;

(* ----- visualization ----- *)

let string_of_op = function
  | Input -> ""
  | Add -> "+"
  | Mul -> "*"
  | Tanh -> "tanh"
  | Pow e -> Printf.sprintf "**%g" e
;;

(* Emit a Graphviz DOT description of the DAG rooted at [root], mirroring
   Karpathy's diagrams: each value is a record box (data | grad) and each op is a
   small oval feeding its result box. Pass the [grads] returned by [backward] to
   annotate the boxes (defaults to 0 when omitted).

   Node identity comes from position in [topo], which is already deduplicated by
   physical identity -- so a value shared by several ops is drawn once, and this
   stays purely functional (no id counter, no emitted-set ref). *)
let to_dot ?(grads = []) root =
  let order = topo root in
  let id n =
    let rec idx i = function
      | x :: _ when x == n -> i
      | _ :: tl -> idx (i + 1) tl
      | [] -> assert false
    in
    idx 0 order
  in
  let node_lines n =
    let i = id n in
    let box =
      Printf.sprintf
        "  n%d [shape=record label=\"data %.4f | grad %.4f\"];\n"
        i
        n.data
        (grad_of grads n)
    in
    let op_box =
      match n.op with
      | Input -> ""
      | _ ->
        Printf.sprintf "  op%d [label=%S];\n  op%d -> n%d;\n" i (string_of_op n.op) i i
    in
    let edges =
      List.map
        (fun c ->
           match n.op with
           | Input -> Printf.sprintf "  n%d -> n%d;\n" (id c) i
           | _ -> Printf.sprintf "  n%d -> op%d;\n" (id c) i)
        n.prev
    in
    box ^ op_box ^ String.concat "" edges
  in
  "digraph G {\n  rankdir=LR;\n"
  ^ String.concat "" (List.map node_lines order)
  ^ "}\n"
;;

(* ----- neural net ----- *)

type neuron = value list * value (* weights, bias *)
type layer = neuron list
type mlp = layer list

(* Weights and bias initialized uniformly in (-1, 1). Randomness is the one
   deliberate effect in this file; seed it from the caller (e.g. [Random.self_init]
   in [main]) so nothing runs at module-load time. *)
let create_neuron (n_in : int) : neuron =
  let random_weight () = input (Random.float 2. -. 1.) in
  List.of_seq (Seq.init n_in (fun _ -> random_weight ())), random_weight ()
;;

(* A neuron computes tanh(w . x + b): the nonlinearity is what lets stacked
   layers represent something a single linear map cannot. *)
let call_neuron (inputs : value list) ((weights, bias) : neuron) : value =
  List.fold_left (fun sum (w, x) -> sum +& (w *& x)) bias (List.combine weights inputs)
  |> tanh
;;

let create_layer (n_in : int) (n_out : int) : layer =
  List.of_seq (Seq.init n_out (fun _ -> create_neuron n_in))
;;

let call_layer (inputs : value list) (l : layer) : value list =
  List.map (call_neuron inputs) l
;;

(* Slide a binary function across consecutive pairs: [a; b; c] -> [f a b; f b c].
   Turns a width list [n_in; h1; ...; n_out] into per-layer (fan-in, fan-out)
   pairs. *)
let rec map_2el f l =
  match l with
  | fst :: (snd :: _ as rest) -> f fst snd :: map_2el f rest
  | _ -> []
;;

let create_mlp (n_in : int) (n_outs : int list) : mlp = map_2el create_layer (n_in :: n_outs)
let call_mlp (inputs : value list) (m : mlp) : value list = List.fold_left call_layer inputs m

(* ----- parameters ----- *)

let neuron_params ((ws, b) : neuron) : value list = b :: ws
let layer_params (l : layer) : value list = List.concat_map neuron_params l
let mlp_params (m : mlp) : value list = List.concat_map layer_params m

(* Rebuild a network, replacing every parameter [p] with [f p]. Purely
   functional: the input [mlp] is untouched and a fresh one is returned. This is
   how a gradient step "updates weights" without mutation. *)
let map_neuron f ((ws, b) : neuron) : neuron = List.map f ws, f b
let map_layer f (l : layer) : layer = List.map (map_neuron f) l
let map_mlp f (m : mlp) : mlp = List.map (map_layer f) m

(* ----- training ----- *)

(* Scalar prediction for one input row (assumes a single output unit). *)
let predict (m : mlp) (xrow : float list) : value =
  match call_mlp (List.map input xrow) m with
  | [ o ] -> o
  | outs ->
    invalid_arg (Printf.sprintf "predict: expected 1 output, got %d" (List.length outs))
;;

(* Sum of squared errors over the dataset -- one scalar [value] whose DAG shares
   the network's parameters across every example, so backprop accumulates each
   parameter's gradient over the whole batch. *)
let loss (m : mlp) (xs : float list list) (ys : float list) : value =
  List.fold_left2
    (fun acc xrow y ->
       let diff = predict m xrow -& input y in
       acc +& (diff **& 2.))
    (input 0.)
    xs
    ys
;;

(* One gradient-descent step. Returns the current loss and a *new* network with
   every parameter nudged down its gradient. Pure: [m] is not modified. *)
let step ~lr (xs : float list list) (ys : float list) (m : mlp) : float * mlp =
  let l = loss m xs ys in
  let g = backward l in
  let m' = map_mlp (fun p -> input (p.data -. (lr *. grad_of g p))) m in
  l.data, m'
;;

(* Run [epochs] gradient-descent steps. Returns the trained network and the loss
   history (oldest first). No I/O -- the caller decides what to print. *)
let train ~lr ~epochs (xs : float list list) (ys : float list) (m : mlp)
  : mlp * float list
  =
  let rec go k m history =
    if k = 0
    then m, List.rev history
    else (
      let l, m' = step ~lr xs ys m in
      go (k - 1) m' (l :: history))
  in
  go epochs m []
;;

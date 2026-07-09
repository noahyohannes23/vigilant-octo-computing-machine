let greet name = Printf.sprintf "hello, %s, from inside the devcontainer" name
let () = print_endline (greet "world")

type ratio =
  { num : int
  ; denom : int
  }

let add_ratio r1 r2 =
  { num = (r1.num * r2.denom) + (r2.num * r1.denom); denom = r1.denom * r2.denom }
;;

let integer_part { num; denom } = num / denom
let get_denom { denom; _ } = denom
let get_num { num; _ } = num
let integer_product integer ratio = { ratio with num = integer * ratio.num }

type number =
  | Int of int
  | Float of float
  | Error

type sign =
  | Positive
  | Negative

let sign_int n = if n >= 0 then Positive else Negative

let add_num a b =
  match a, b with
  | Int ia, Int ib ->
    if sign_int ia = sign_int ib && sign_int (ia + ib) <> sign_int ia
    then Float (float ia +. float ib)
    else Int (ia + ib)
  | Int ia, Float fb -> Float (float ia +. fb)
  | Float fa, Int ib -> Float (fa +. float ib)
  | Float fa, Float fb -> Float (fa +. fb)
  | Error, _ -> Error
  | _, Error -> Error
;;

type 'a btree =
  | Empty
  | Node of 'a * 'a btree * 'a btree

let rec str_of_btree = function
  | Empty -> "Empty"
  | Node (x, left, right) ->
    Printf.sprintf "Node (%d, %s, %s)" x (str_of_btree left) (str_of_btree right)
;;

let rec member x btree =
  match btree with
  | Empty -> false
  | Node (n, left, right) ->
    if n = x then true else if x < n then member x left else member x right
;;

let rec height btree =
  match btree with
  | Empty -> 0
  | Node (_, left, right) -> 1 + max (height left) (height right)
;;

let balance_factor btree =
  match btree with
  | Empty -> 0
  | Node (_, left, right) -> height right - height left
;;

let balance btree =
  let bf_x = balance_factor btree in
  match btree with
  | Node (x, t1, Node (z, t23, t4))
    when bf_x > 1 && Node (z, t23, t4) |> balance_factor >= 0 ->
    Node (z, Node (x, t1, t23), t4)
  | Node (x, Node (z, t4, t23), t1)
    when bf_x < -1 && Node (z, t4, t23) |> balance_factor <= 0 ->
    Node (z, t4, Node (x, t23, t1))
  | Node (x, t1, Node (z, Node (y, t2, t3), t4))
    when bf_x > 1 && Node (z, Node (y, t2, t3), t4) |> balance_factor < 0 ->
    Node (y, Node (x, t1, t2), Node (z, t3, t4))
  | Node (x, Node (z, t4, Node (y, t3, t2)), t1)
    when bf_x < -1 && Node (z, t4, Node (y, t3, t2)) |> balance_factor > 0 ->
    Node (y, Node (z, t4, t3), Node (x, t2, t1))
  | _ -> btree
;;

let rec insert_unbalanced x btree =
  match btree with
  | Empty -> Node (x, Empty, Empty)
  | Node (n, left, right) when n = x -> Node (n, left, right)
  | Node (n, left, right) ->
    if x < n
    then Node (n, insert_unbalanced x left, right)
    else Node (n, left, insert_unbalanced x right)
;;

let rec insert x btree =
  match btree with
  | Empty -> Node (x, Empty, Empty)
  | Node (n, left, right) when n = x -> Node (n, left, right)
  | Node (n, left, right) ->
    balance
      (if x > n then Node (n, left, insert x right) else Node (n, insert x left, right))
;;

let%test _ =
  let n = 3 in
  Empty |> insert n = Node (n, Empty, Empty)
;;

let%test _ = Empty |> insert_unbalanced 3 |> insert_unbalanced 3 = Node (3, Empty, Empty)

let%test _ =
  Empty
  |> insert_unbalanced 3
  |> insert_unbalanced 1
  |> insert_unbalanced 4
  = (Empty |> insert_unbalanced 3 |> insert_unbalanced 4 |> insert_unbalanced 1)
;;

let tree =
  Node
    ( 50
    , Node (20, Node (10, Empty, Empty), Node (30, Empty, Empty))
    , Node (100, Empty, Empty) )
;;

let%test _ = balance tree = tree
let%test _ = member 10 tree
let%test _ = member 20 tree
let%test _ = member 30 tree
let%test _ = member 100 tree
let%test _ = false = member 0 tree
let%test _ = false = member 23 tree
let%test _ = false = member 101 tree
let%test _ = height tree = 3
let%test _ = tree |> insert 75 |> insert 200 |> height = height tree

let%test _ =
  let t2 = Empty |> insert 20 |> insert 50 |> insert 100 |> insert 30 |> insert 10 in
  t2 = tree
;;

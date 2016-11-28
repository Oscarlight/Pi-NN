open Printf
open Str

(* ================================================================= *
 * Utility functions for array
 * ================================================================= *)

module Array = struct
  include Array
  (* val init_matrix : int -> int -> (int -> int -> 'a) -> 'a array array *)
  let init_matrix m n f = init m (fun i -> init n (f i))
  (* val matrix_size : 'a array array -> int * int *)
  let matrix_size a =
    let m = length a in
    let n = if m = 0 then 0 else length a.(0) in
    (m, n)
  (* val map2 : ('a -> 'b -> 'c) -> 'a array -> 'b array -> 'c array *)
  let map2 f x y = mapi (fun i xi -> f xi y.(i)) x
  (* val iter2 : ('a -> 'b -> unit) -> 'a array -> 'b array -> unit *)
  let iter2 f x y = iteri (fun i xi -> f xi y.(i)) x
  (* val iter3 : ('a -> 'b -> 'c -> unit) -> 
     'a array -> 'b array -> 'c array -> unit *)
  let iter3 f x y z = iteri (fun i xi -> f xi y.(i) z.(i)) x
  (* val iter4 : ('a -> 'b -> 'c -> 'd -> unit) -> 
     'a array -> 'b array -> 'c array -> 'd array -> unit *)
  let iter4 f g x y z = iteri (fun i xi -> f g.(i) xi y.(i) z.(i)) x
  (* val iteri2 : (int -> 'a -> 'b -> unit) -> 'a array -> 'b array -> unit *)
  let iteri2 f x y = iteri (fun i xi -> f i xi y.(i)) x
  (* val fold_left2 : ('a -> 'b -> 'c -> 'a) -> 'a -> 'b array -> 'c array -> 'a *)
  let fold_left2 f init x y =
    let acc = ref init in
    for i = 0 to length x - 1 do acc := f !acc x.(i) y.(i) done;
    !acc
  (* val map_sum : ('a -> float) -> 'a array -> float *)
  let map_sum f = fold_left (fun acc xi -> acc +. f xi) 0.0
  (* val map2_sum : ('a -> 'b -> float) -> 'a array -> 'b array -> float *)
  let map2_sum f = fold_left2 (fun acc xi yi -> acc +. f xi yi) 0.0
end

(* ================================================================= *
 * BLAS-like functions for linear algebraic operations
 * ================================================================= *)
(** Dot product of two vectors *)
(* val dot : float array -> float array -> float *)
let dot = Array.map2_sum ( *. )
(** [z = x * y], x and y are array, z[i] = x[i] * y[i] *)
let xy = Array.map2 ( *. )
(** [z = x + y], x are array, z[i] = x[i] + y[i] *)
let xpy = Array.map2 ( +. )
(** Execute [y := alpha * x + y] where [alpha] is a scalar, [x] and [y] are
    vectors. *)
(* val axpy : alpha:float -> float array -> float array -> unit *)
let axpy ~alpha x y =
  let n = Array.length x in
  for i = 0 to n - 1 do y.(i) <- alpha *. x.(i) +. y.(i) done
(** Execute [y := alpha * x + y + beta * z] where [alpha], [beta] is a scalar, 
    [x] and [y] are vectors. *)
(* val axpypbz : alpha:float -> beta:float -> float array
  -> float array -> float array -> unit *)
let axpypbz ~alpha ~beta x y z =
  let n = Array.length x in
  for i = 0 to n - 1 do y.(i) <- alpha *. x.(i) +. y.(i) +. beta *. z.(i) done
(** Execute [y := alpha * g * x + y + beta * z] where [alpha], [beta] is a scalar, 
  [g], [x] and [y] are vectors. Use for adagrad *)
(* val agxpypbz : float -> float -> float -> float array
  -> float array -> float array -> float array -> unit *)
let agxpypbz alpha beta gamma g x y z =
  let n = Array.length x in
  for i = 0 to n - 1 do
   if (x.(i) *. z.(i)) > 0.0 then g.(i) <- min 10.0 (g.(i) +. gamma);
   if (x.(i) *. z.(i)) < 0.0 then g.(i) <- max 1e-6 (g.(i) *. (1.0 -. gamma));
   y.(i) <- alpha *. g.(i) *. x.(i) +. y.(i) +. beta *. z.(i) 
  done
(** [gemv a x y] computes [a * x + y] where [a] is a matrix, and [x] and [y] are
    vectors. *)
(* val gemv : float array array -> float array -> float array -> float array *)
let gemv a x y = Array.map2 (fun ai yi -> dot ai x +. yi) a y
(** [ax a x] computes [a * x] where [a] is a matrix, and [x] is vector. *)
(* val gemv : float array array -> float array -> float array -> float array *)
let ax a x  = Array.map (fun ai -> dot ai x) a
(** [gemv_t a x] computes [a^T * x] where [a] is a matrix and [x] is a vector.*)
(* val gemv_t : float array array -> float array -> float array *)
let gemv_t a x =
  let (_, n) = Array.matrix_size a in
  let y = Array.make n 0.0 in
  Array.iter2 (fun ai xi -> axpy ~alpha:xi ai y) a x;
  y
(** [ger x y] computes outer product [x y^T] of vectors [x] and [y]. *)
(* val ger : float array -> float array -> float array array *)
let ger x y = Array.map (fun xi -> Array.map (( *. ) xi) y) x
(** [copy_vec] copy vector x to vector y *)
(* val copy_vec: float array -> float array -> unit *)
let copy_vec x y = Array.iteri (fun i xi -> (y.(i) <- xi)) x
(** [copy_mat] copy matrix x to matrix y *)
(* val copy_mat: float array array -> float array array -> unit *)
let copy_mat m1 m2 = Array.iter2 (copy_vec) m1 m2



(* ================================================================= *
 * Utility functions for write and read matrix and vector
 * ================================================================= *)
let comma = Str.regexp ","

let write_line oc al = 
  let l = Array.to_list al in 
	let rec helper l = match l with
	| [] -> failwith "write_mat: empty row"
	| [a] -> fprintf oc "%.20g\n" a;
	| hd :: tl ->  fprintf oc "%.20g," hd; helper tl 
  in
	helper l


let write_mat filename mat = 
  let oc = open_out (filename ^ ".csv") in
  	Array.iter (write_line oc) mat; close_out oc;;

let write_vec filename vec =
  let oc = open_out (filename ^ ".csv") in
  	write_line oc vec; close_out oc;;


let read_mat filename m n =
  let mat = Array.make_matrix m n 0.0 in
  let i = ref 0 in
  try
  	let ic = open_in (filename ^ ".csv") in
  	try
  	  while true; do 
  	    let line_list = Str.split comma (input_line ic) in
  	    assert(List.length line_list = n);
  	    let line_float_list = List.map (float_of_string) line_list in
  	    let line_array = Array.of_list line_float_list in
  	      mat.(!i) <- line_array;
  	      i := !i + 1;
  	  done;
  	  mat
  	with
  	| End_of_file -> assert(m = !i); close_in ic; mat
  with
  | e -> raise e;;
  

let read_vec filename n =
  let vec = ref [||] in
  try
  	let ic = open_in (filename ^ ".csv") in
  	try 
  	  let line_list = Str.split comma (input_line ic) in
  	  assert(List.length line_list = n);
  	  let line_float_list = List.map (float_of_string) line_list in
  	  let line_array = Array.of_list line_float_list in
  	  vec := line_array;
  	  !vec
  	with
  	| End_of_file -> close_in ic; !vec
  with
  | e -> raise e;;


let appendFile filename line =
  let out = open_out_gen [Open_wronly; Open_append; Open_creat; Open_text] 0o666 filename in
    output_string out line;
    close_out out;;



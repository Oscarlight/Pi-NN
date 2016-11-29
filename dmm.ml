(** feed forward neutral network for device compact modeling
    Mingda Li (Nov. 29, 2016)
    Inspired by the 2015 Akinori ABE *)

open Format
open Inout
open Util

(* ================================================================= *
 * Error functions and its Direvative
 * ================================================================= *)
(** quadratic functions *)
let error_st y t = (Array.map2_sum (fun yi ti -> 0.5 *. (ti -. yi) ** 2. ) y t)
let error_st' = Array.map2 (fun yi ti -> yi -. ti)

(* ================================================================= *
 * Multilayer neural network
 * ================================================================= *)
(** A layer in a multilayer neural network *)
type layer =
  {
    (* source-drain layer (i.e. tanh network) *)
    actv_f1 : float array -> float array; (** an activation function *)
    actv_f1' : float array -> float array array; (** the derivative of [actv_f]*)
    weight1 : float array array; (** a weight matrix *)
    bias1 : float array; (** a bias vector *) 
    (* gating layer (i.e. sigmoid network) *)
    actv_f2 : float array -> float array; (** an activation function *)
    actv_f2' : float array -> float array array; (** the derivative of [actv_f]*)
    weight2 : float array array; (** a weight matrix *)
    bias2 : float array; (** a bias vector *)
    (* from source-drain layer to gating layer *)
    iweight : float array array; (* inter-network weights *)
    (* for momentum & adagrad *)
    weight_c1 : float array array; (** a previous weight change matrix *)
    (* bias_c1 : float array; (* a bias change vector *)  *)
    gain_w1 : float array array; (* local weight learning rate gain matrix *)
    (* gain_b1 : float array; local bias learning rate gain vector *)
    weight_c2 : float array array; (** a previous weight change matrix *)
    bias_c2 : float array; (** a bias change vector *)
    gain_w2 : float array array; (* local weight learning rate gain matrix *)
    gain_b2 : float array; (* local bias learning rate gain vector *)
    iweight_c : float array array;
    gain_iw : float array array;
  }

(** Forward propagation *)
(* val forwardprop : layer list -> (float array * float array) -> float array = <fun> *)
let forwardprop lyrs x0_pair =
  let (yo1, yo2) = List.fold_left
    (fun (xi1,xi2) lyr -> let y1 = lyr.actv_f1 (gemv lyr.weight1 xi1 lyr.bias1) in
    	let y2 = lyr.actv_f2 (xpy (ax lyr.iweight y1) (gemv lyr.weight2 xi2 lyr.bias2)) in
    	(y1, y2)) x0_pair lyrs in
  xy yo1 yo2

(** Error backpropagation *)
(* val backprop : layer list -> float array * float array -> float array -> 
   (float array * float array) list *)
let backprop1 lyrs (x1_0, x2_0) t =
  let rec calc_delta (x1, x2) = function
    | [] -> failwith "empty neural network"
    | [lyr] -> (* output layer *)
      let y1 = lyr.actv_f1 (gemv lyr.weight1 x1 lyr.bias1) in
      let y2 = lyr.actv_f2 (xpy (ax lyr.iweight y1) (gemv lyr.weight2 x2 lyr.bias2)) in
      let err = error_st' (xy y1 y2) t in
      let delta1 = gemv_t (lyr.actv_f1' y1) (xy y2 err) in
      (delta1, [])
    | lyr :: ((uplyr :: _) as lyrs') -> (* hidden layer *)
      let y1 = lyr.actv_f1 (gemv lyr.weight1 x1 lyr.bias1) in
      let y2 = lyr.actv_f2 (xpy (ax lyr.iweight y1) (gemv lyr.weight2 x2 lyr.bias2)) in
      let (updelta1, tl) = calc_delta (y1, y2) lyrs' in
      let delta1 = gemv_t (lyr.actv_f1' y1) (gemv_t uplyr.weight1 updelta1) in
      (delta1, (y1, updelta1) :: tl)
  in
  let (delta1_0, tl) = calc_delta (x1_0, x2_0) lyrs in
  (x1_0, delta1_0) :: tl

(* val backprop : layer list -> float array * float array -> float array -> 
   (float array * float array) * (float array * float array * float array * float array * float array) list *)
let backprop2 lyrs (x1_0, x2_0) t = 
  let rec calc_delta (x1, x2) = function
  	| [] -> failwith "empty neural network"
  	| [lyr] -> (* output layer *)
  	  let y1 = lyr.actv_f1 (gemv lyr.weight1 x1 lyr.bias1) in
      let y2 = lyr.actv_f2 (xpy (ax lyr.iweight y1) (gemv lyr.weight2 x2 lyr.bias2)) in
      let err = error_st' (xy y1 y2) t in
      (* add log err *)
      (* let logerr = error_log' (xy y1 y2) t in *)
      let delta2 = gemv_t (lyr.actv_f2' y2) (xy y1 err) in
      let delta1 = gemv_t (lyr.actv_f1' y1) (xpy (xy y2 err) (gemv_t lyr.iweight delta2)) in
      (y1, delta1, delta2, [])
    | lyr :: ((uplyr :: _) as lyrs') -> (* hidden layer *)
      let y1 = lyr.actv_f1 (gemv lyr.weight1 x1 lyr.bias1) in
      let y2 = lyr.actv_f2 (xpy (ax lyr.iweight y1) (gemv lyr.weight2 x2 lyr.bias2)) in   
	  let (upy1, updelta1, updelta2, tl) = calc_delta (y1, y2) lyrs' in
	  let delta2 = gemv_t (lyr.actv_f2' y2) (gemv_t uplyr.weight2 updelta2) in
	  let delta1 = gemv_t (lyr.actv_f1' y1) (xpy (gemv_t uplyr.weight1 updelta1) (gemv_t lyr.iweight delta2)) in
	  (y1, delta1, delta2, (y1, updelta1, y2, updelta2, upy1) :: tl)
	in
	let (y1_0, delta1_0, delta2_0, tl) = calc_delta (x1_0, x2_0) lyrs in
	(x1_0, delta1_0, x2_0, delta2_0, y1_0) :: tl

(** Update parameters in the given neural network according to the given input
    and target (stochastic gradient descent). *)
(* val train : eta:float -> layer list -> float array -> float array -> unit *)
let train2 ~eta ~m ~g lyrs (x1_0, x2_0) t =
  (* printf "-->> Flag: %s \n" "1"; *)
  let res = backprop2 lyrs (x1_0, x2_0) t in
   List.iter2
    (fun (x1, updelta1, x2, updelta2, upy1) lyr ->
       let dw1 = ger updelta1 x1 in
       let dw2 = ger updelta2 x2 in
       let diw = ger updelta2 upy1 in
       (* let db1 = updelta1 in *) (* no bias *)
       let db2 = updelta2 in
   	   Array.iter4 (agxpypbz (~-. eta) (~-. m) g) lyr.gain_w1 dw1 lyr.weight1 lyr.weight_c1;
   	   (* printf "-->> Flag: %s \n" "6"; *)
   	   Array.iter4 (agxpypbz (~-. eta) (~-. m) g) lyr.gain_w2 dw2 lyr.weight2 lyr.weight_c2;
   	   (* printf "-->> Flag: %s \n" "7"; *)
   	   Array.iter4 (agxpypbz (~-. eta) (~-. m) g) lyr.gain_iw diw lyr.iweight lyr.iweight_c;
   	   (* printf "-->> Flag: %s \n" "5"; *)
   	   (* agxpypbz (~-.eta) (~-.m) g lyr.gain_b1 db1 lyr.bias1 lyr.bias_c1; *) (* no bias *)
       agxpypbz (~-.eta) (~-.m) g lyr.gain_b2 db2 lyr.bias2 lyr.bias_c2;
  	   copy_mat dw1 lyr.weight_c1; copy_mat dw2 lyr.weight_c2; copy_mat diw lyr.iweight_c; 
       (* copy_vec db1 lyr.bias_c1;  *) (* no bias *)
       copy_vec db2 lyr.bias_c2;
       (* printf "-->> Flag: %s \n" "4"; *)
	) res lyrs


(* ================================================================= *
 * Activation functions 
 * ================================================================= *)
(** The hyperbolic tangent *)
let actv_tanh = Array.map (fun x -> (tanh x))
(** The derivative of the hyperbolic tangent *)
let actv_tanh' z =
  let n = Array.length z in
  Array.init_matrix n n (fun i j -> if i=j then  1.0 -. z.(i) *. z.(i) else 0.0)
(** sigmoid-shaped activation function: logistic function *)
(* float array -> float array *)
let actv_sigmoid = Array.map (fun x -> (1.0 /. (1.0 +. exp(-. x)) ))
(* float array -> float array array *)
let actv_sigmoid' z = 
  let n = Array.length z in
  Array.init_matrix n n (fun i j -> if i=j then z.(i) *. (1.0 -. z.(i)) else 0.0)

(* ================================================================= *
 * Main routine
 * ================================================================= *)

(** Return a layer of a neural network. *)
(* val make_layer : (float array -> float array) -> 
  (float array -> float array array) -> int -> int -> layer *)
let make_layer actv_f1 actv_f1' actv_f2 actv_f2' dim1 updim1 dim2 updim2 rlow rup =
  let rand () = rlow +. Random.float (rup -. rlow) in
  { (* source-drain layer *)
  	actv_f1; actv_f1';
    weight1 = Array.init_matrix updim1 dim1 (fun _ _ -> rand ());
    (* bias1 = Array.init updim1 (fun _ -> rand ()); *)
    bias1 = Array.init updim1 (fun _ -> 0.); (* no bias *)
    (* gating layer *)
    actv_f2; actv_f2';
    weight2 = Array.init_matrix updim2 dim2 (fun _ _ -> rand ());
    bias2 = Array.init updim2 (fun _ -> rand ());
    (* from source-drain layer to gating layer *)
    iweight = Array.init_matrix updim2 updim1 (fun _ _ -> rand ());
    (* for momentum & adagrad *)
    weight_c1 = Array.init_matrix updim1 dim1 (fun _ _ -> 0.);
    (* bias_c1 = Array.init updim1 (fun _ -> 0.); *)
    gain_w1 = Array.init_matrix updim1 dim1 (fun _ _ -> 1.);
    (* gain_b1 = Array.init updim1 (fun _ -> 1.); *)
    weight_c2 = Array.init_matrix updim2 dim2 (fun _ _ -> 0.);
    bias_c2 = Array.init updim2 (fun _ -> 0.);
    gain_w2 = Array.init_matrix updim2 dim2 (fun _ _ -> 1.);
    gain_b2 = Array.init updim2 (fun _ -> 1.);
    iweight_c = Array.init_matrix updim2 updim1 (fun _ _ -> 0.);
    gain_iw = Array.init_matrix updim2 updim1 (fun _ _ -> 1.)}


(** Evaluate an error *)
(* val evaluate : layer list -> (float array * float array) array -> float *)
let evaluate error_func lyrs samples =
  Array.map_sum (fun (x_s, x_t, t) -> 
    error_func (forwardprop lyrs (x_s, x_t)) t) samples

(* val main : (float array * float array) array -> bytes -> unit *)
let main samples sampleTests dir=
  let savedir = dir in
  printf "-->> save to: %s \n" savedir;
  (* Build the neural network *)
  let hidden_dim_1 = 2 in 
  let hidden_dim_2 = 3 in
  printf "hidden_dim_1: %d; hidden_dim_2: %d \n" hidden_dim_1 hidden_dim_2;
  let nnet = [
    make_layer actv_tanh actv_tanh' actv_sigmoid actv_sigmoid' 
    		   1 hidden_dim_1 1 hidden_dim_2 (-0.2) 0.4;
    make_layer actv_tanh actv_tanh' actv_sigmoid actv_sigmoid' 
    		   hidden_dim_1 1 hidden_dim_2 1 (-2.0) (4.0);
  ] in
  (* Training *)
  let (eta, m, g) = (0.005, 0.05, 0.001) in
  let () = printf "Learning: eta: %g; m: %g; gain: %g\n" eta m g in
  printf "Initial: Standard Error = %g@." (evaluate error_st nnet samples);
  for i = 1 to 5000000 do
    Array.iter (fun (x_s, x_t, t) ->
        (* check_gradient nnet x t; *)
        train2 ~eta ~m ~g nnet (x_s, x_t) t) samples;
    if i mod 10000 = 0 then 
    begin
      (* save *)
      let j = ref 0 in
      ignore (List.map (fun lyr -> 
        write_mat (savedir^"/w1_"^(string_of_int !j)) lyr.weight1;
        write_vec (savedir^"/b1_"^(string_of_int !j)) lyr.bias1;
        write_mat (savedir^"/w2_"^(string_of_int !j)) lyr.weight2;
        write_vec (savedir^"/b2_"^(string_of_int !j)) lyr.bias2;
        write_mat (savedir^"/iw_"^(string_of_int !j)) lyr.iweight;
        j := !j + 1;) nnet);
      let trainErr = evaluate error_st nnet samples in
      let testErr = evaluate error_st nnet sampleTests in
        printf "Loop #%d: trainErr = %g, testErr = %g@." i trainErr testErr;
        appendFile (savedir^"/log") ((string_of_int i)^", "^(string_of_float trainErr)
                          ^", "^(string_of_float testErr)^"\n");
    end
  done;;


let filename = "data/Thin_TFET_train" (* training file dir *) 
let fileTest = "data/Thin_TFET_test" (* testing file dir *) 

let () = printf "Input file is: %s\n" filename
(* ([|Vds|], [|Vtg|], [|Id|]) *)
let patts = import2 (filename^".csv") "," 2091 (* # of training cases *) 
let testPatts = import2 (fileTest^".csv") "," 2000 (* # of testing cases *)
(* num_case must match! or add wrong cases*)
let () = main patts testPatts "model"; (* output dir *)


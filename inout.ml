open Printf
open Str

(* import: (x, t) *)
let import file_name separator num_case =
  let reg_separator = Str.regexp separator in
  let i = ref 0 in
  try
    let ic = open_in file_name in
    (* Skip the first line, columns headers *)
    let fst_line = Str.split reg_separator (input_line ic) in
      let len = List.length fst_line in
      Printf.printf "Num of columns: %d \n" len;
        let value_array = Array.make num_case 
          (Array.make (len - 1) 0.0, [|0.0|]) in
    try
      while !i < num_case do
        (* Create a list of values from a line *)
        let line_list = Str.split reg_separator (input_line ic) in
        let rec add ll (ilst, olst) =
          match ll with
          | h1::h2::tl -> add (h2::tl) 
            ((float_of_string h1)::ilst, olst)
          | hd::tl -> (ilst, (float_of_string hd)::olst)
          | [] -> failwith "Need more than 1 columns" in
        let (ilst, olst) = add line_list ([], []) in
          value_array.(!i) <- (Array.of_list ilst, Array.of_list olst);
        i := !i + 1;
        (* Printf.printf "Loading: %d \n" !i; *)
      done;
      Printf.printf "Num. of training case (before end): %d \n" !i;
      value_array
    with 
      | End_of_file -> Printf.printf "Num. of training case: %d \n" !i;
                       close_in ic; value_array
   with
     | e -> raise e;;

(* import2: (x1, x2, t) *)
let import2 file_name separator num_case =
  let reg_separator = Str.regexp separator in
  let i = ref 0 in
  try
    let ic = open_in file_name in
    (* Skip the first line, columns headers *)
    let fst_line = Str.split reg_separator (input_line ic) in
      let len = List.length fst_line in
      Printf.printf "Num of columns: %d \n" len;
        let value_array = Array.make num_case 
          ([|0.0|], [|0.0|], [|0.0|]) in (* special case, x0_s and x0_t has only one elements*)
    try
      while !i < num_case do
        (* Create a list of values from a line *)
        let line_list = Str.split reg_separator (input_line ic) in
        (* Special case for ThinTFET *) 
        value_array.(!i) <- (Array.make 1 (float_of_string (List.nth line_list 1)), (* Vds *) 
                             Array.make 1 (float_of_string (List.nth line_list 0)), (* Vtg *)
                             Array.make 1 (float_of_string (List.nth line_list 2)));
        i := !i + 1;
        (* Printf.printf "Loading: %d \n" !i; *)
      done;
      Printf.printf "Num. of training case (before end): %d \n" !i;
      value_array
    with 
      | End_of_file -> Printf.printf "Num. of training case: %d \n" !i;
                       close_in ic; value_array
   with
     | e -> raise e;;

(* testing *)
let print_array ff print xs =
  let n = Array.length xs in
    if n=0 then fprintf ff "[||]" else begin
      fprintf ff "[|";
      for i=0 to Array.length xs-2 do
        fprintf ff "%a; " print xs.(i)
      done;
      fprintf ff "%a|]" print xs.(n-1)
    end;;

(* print the output of the neural network for each input *)
let test_print patts net run =
  let aux (inputs, _) =
    let print ff = print_array ff (fun ff -> fprintf ff "%g") in
    let outputs = run net inputs in
    printf "%a -> %a\n" print inputs print outputs in
  Array.iter aux patts
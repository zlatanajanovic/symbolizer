(define (problem problem_15_problem_23_state_0_domain_cooking_10)
  (:domain cooking)
  (:objects
    carrot_pieces tomato - vegetable
    a_bot b_bot - robot
    knife - tool
    counter bowl cutting_board - location
    )
  (:init
    (is-sliced carrot_pieces)
    (is-workspace cutting_board)
    (can-cut knife)
    (available tomato)
    (at tomato counter)
    (free a_bot)
    (carry b_bot knife)
    )
  (:goal
    (and (at carrot_pieces bowl) (at tomato bowl) (is-sliced tomato) )))
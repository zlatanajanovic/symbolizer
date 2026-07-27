(define (problem problem_5_problem_25_state_0_domain_cooking_2)
  (:domain cooking)
  (:objects
    tomato - vegetable
    a_bot b_bot - robot
    knife - tool
    counter bowl cutting_board - location
    )
  (:init
    (is-whole tomato)
    (is-workspace cutting_board)
    (can-cut knife)
    (available tomato)
    (carry b_bot knife)
    (at tomato counter)
    (free a_bot)
    )
  (:goal
    (and (at tomato bowl) (is-sliced tomato) )))
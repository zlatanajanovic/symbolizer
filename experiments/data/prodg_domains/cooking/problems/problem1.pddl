(define (problem problem_26_problem_7_state_0_domain_cooking_1)
  (:domain cooking)
  (:objects
    a_bot b_bot - robot
    knife - tool
    cucumber - vegetable
    counter bowl cutting_board - location
    )
  (:init
    (is-whole cucumber)
    (is-workspace cutting_board)
    (can-cut knife)
    (available cucumber)
    (at cucumber counter)
    (free a_bot)
    )
  (:goal
    (and (at cucumber bowl) (is-sliced cucumber) )))
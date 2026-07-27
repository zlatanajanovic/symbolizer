(define (problem problem_12_problem_21_state_0_domain_blocksworld_1)
  (:domain blocksworld_with_robot)
  (:objects
    green_block yellow_block pink_block red_block - block
    robot_1 - robot
    )
  (:init
    (ontable green_block)
    (ontable yellow_block)
    (ontable pink_block)
    (ontable red_block)
    (clear green_block)
    (clear yellow_block)
    (clear pink_block)
    (clear red_block)
    (handempty robot_1)
    )
  (:goal
    (and (on pink_block red_block) (on red_block yellow_block) (on yellow_block green_block) (clear pink_block) )))
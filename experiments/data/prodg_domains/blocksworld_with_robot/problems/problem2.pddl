(define (problem problem_29_problem_1_state_0_domain_blocksworld_2)
  (:domain blocksworld_with_robot)
  (:objects
    pink_block purple_block blue_block red_block - block
    robot_1 - robot
    )
  (:init
    (ontable blue_block)
    (on purple_block blue_block)
    (on pink_block purple_block)
    (ontable red_block)
    (clear pink_block)
    (clear red_block)
    (holding red_block)
    (handfull robot_1)
    )
  (:goal
    (and (on red_block blue_block) (on blue_block purple_block) (on purple_block pink_block) )))
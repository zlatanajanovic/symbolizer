(define (problem problem_1_problem_29_state_0_domain_hanoi_1)
  (:domain hanoi_with_types)
  (:objects
    orange_disk1 peg2 peg1 peg3 blue_disk1 green_disk1 pink_disk1 - element
    )
  (:init
    (on pink_disk1 green_disk1)
    (on green_disk1 blue_disk1)
    (on blue_disk1 orange_disk1)
    (on orange_disk1 peg1)
    (clear pink_disk1)
    (clear peg2)
    (clear peg3)
    (smaller blue_disk1 orange_disk1)
    (smaller green_disk1 orange_disk1)
    (smaller pink_disk1 orange_disk1)
    (smaller green_disk1 blue_disk1)
    (smaller pink_disk1 blue_disk1)
    (smaller pink_disk1 green_disk1)
    )
  (:goal
    (and (on orange_disk1 peg3) (on blue_disk1 orange_disk1) (on green_disk1 blue_disk1) (on pink_disk1 green_disk1) (clear pink_disk1) (clear peg1) (clear peg2) (smaller blue_disk1 orange_disk1) (smaller green_disk1 blue_disk1) (smaller pink_disk1 green_disk1) )))
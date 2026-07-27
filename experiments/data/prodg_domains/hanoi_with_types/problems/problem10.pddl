(define (problem problem_10_problem_26_state_0_domain_hanoi_10)
  (:domain hanoi_with_types)
  (:objects
    orange_disk1 pink_disk1 yellow_disk1 green_disk1 blue_disk1 peg1 peg2 peg3 - element
    )
  (:init
    (on blue_disk1 pink_disk1)
    (on pink_disk1 green_disk1)
    (on green_disk1 yellow_disk1)
    (on yellow_disk1 orange_disk1)
    (on orange_disk1 peg1)
    (clear blue_disk1)
    (clear peg2)
    (clear peg3)
    (smaller blue_disk1 pink_disk1)
    (smaller pink_disk1 green_disk1)
    (smaller green_disk1 yellow_disk1)
    (smaller yellow_disk1 orange_disk1)
    )
  (:goal
    (and (on orange_disk1 peg3) )))
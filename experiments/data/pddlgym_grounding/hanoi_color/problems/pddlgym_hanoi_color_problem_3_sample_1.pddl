
(define (problem hanoiZero5) (:domain hanoi)
  (:objects
        red_disk - disc_or_peg
	green_disk - disc_or_peg
	blue_disk - disc_or_peg
	left_peg - disc_or_peg
	middle_peg - disc_or_peg
	right_peg - disc_or_peg
  )
(:init
	(clear red_disk)
	(clear green_disk)
	(clear middle_peg)
	(on red_disk left_peg)
	(on green_disk blue_disk)
	(on blue_disk right_peg)
	(smaller green_disk red_disk)
	(smaller blue_disk red_disk)
	(smaller blue_disk green_disk)
	(smaller left_peg red_disk)
	(smaller left_peg green_disk)
	(smaller left_peg blue_disk)
	(smaller middle_peg red_disk)
	(smaller middle_peg green_disk)
	(smaller middle_peg blue_disk)
	(smaller right_peg red_disk)
	(smaller right_peg green_disk)
	(smaller right_peg blue_disk)
)
(:goal (and
	(on blue_disk right_peg)
	(on green_disk blue_disk)
	(on red_disk green_disk)))
)
  

(define (problem hanoi1)
    (:domain hanoi_with_types)

    (:objects
        peg1 - element
        peg2 - element
        peg3 - element
        green_disk1 - element
        blue_disk1 - element
        pink_disk1 - element
    )
        (:init
	(clear green_disk1)
	(clear peg2)
	(clear peg3)
	(on blue_disk1 pink_disk1)
	(on green_disk1 blue_disk1)
	(on pink_disk1 peg1)
	(smaller blue_disk1 green_disk1)
	(smaller peg1 blue_disk1)
	(smaller peg1 green_disk1)
	(smaller peg1 pink_disk1)
	(smaller peg2 blue_disk1)
	(smaller peg2 green_disk1)
	(smaller peg2 pink_disk1)
	(smaller peg3 blue_disk1)
	(smaller peg3 green_disk1)
	(smaller peg3 pink_disk1)
	(smaller pink_disk1 blue_disk1)
	(smaller pink_disk1 green_disk1)
)
    (:goal (and (on pink_disk1 peg3) (on blue_disk1 pink_disk1) (on green_disk1 blue_disk1)))
)
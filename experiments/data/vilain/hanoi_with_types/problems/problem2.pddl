(define (problem hanoi2)
    (:domain hanoi_with_types)
    (:objects
        peg1 - element
        peg2 - element
        peg3 - element
        purple_disk1 - element
        blue_disk1 - element
        pink_disk1 - element
        orange_disk1 - element
    )
        (:init
            (clear purple_disk1)
            (clear peg2)
            (clear peg3)
            (on purple_disk1 blue_disk1)
            (on blue_disk1 pink_disk1)
            (on pink_disk1 orange_disk1)
            (on orange_disk1 peg1)
            (smaller blue_disk1 purple_disk1)
            (smaller pink_disk1 purple_disk1)
            (smaller pink_disk1 blue_disk1)
            (smaller orange_disk1 purple_disk1)
            (smaller orange_disk1 blue_disk1)
            (smaller orange_disk1 pink_disk1)
            (smaller peg1 purple_disk1)
            (smaller peg1 blue_disk1)
            (smaller peg1 pink_disk1)
            (smaller peg1 orange_disk1)
            (smaller peg2 purple_disk1)
            (smaller peg2 blue_disk1)
            (smaller peg2 pink_disk1)
            (smaller peg2 orange_disk1)
            (smaller peg3 purple_disk1)
            (smaller peg3 blue_disk1)
            (smaller peg3 pink_disk1)
            (smaller peg3 orange_disk1)
        )
    (:goal (and (on orange_disk1 peg3) (on pink_disk1 orange_disk1) (on blue_disk1 pink_disk1) (on purple_disk1 blue_disk1)))
)
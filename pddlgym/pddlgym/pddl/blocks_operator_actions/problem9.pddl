(define (problem blocks)
    (:domain blocks)
    (:objects 
        yellow - block
        red - block
        green - block
        grey - block
        blue - block
        purple - block
    )
    (:init
        (clear red)
        (on red grey)
        (on grey yellow)
        (on yellow blue)
        (on blue purple)
        (ontable purple)
        (holding green)
        (handfull)

    )
    (:goal (and (on red grey) (on grey yellow) (on yellow blue) (on blue purple) (on purple green)))
)
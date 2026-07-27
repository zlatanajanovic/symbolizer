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
        (clear green)
        (on green red)
        (on red grey)
        (ontable grey)
        (ontable blue)
        (clear blue)
        (holding yellow)
        (ontable purple)
        (clear purple)
        (handfull)

    )
    (:goal (and (on green red) (on red grey) (on grey yellow) (on yellow blue) (on blue purple)))
)

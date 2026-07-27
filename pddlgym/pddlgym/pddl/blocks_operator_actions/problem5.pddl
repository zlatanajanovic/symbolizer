(define (problem blocks)
    (:domain blocks)
    (:objects 
        purple - block
        grey - block
        green - block
        blue - block
        yellow - block
    )
    (:init
        (clear green)
        (on green grey)
        (on grey blue)
        (on blue purple)
        (on purple yellow)
        (ontable yellow)
        (handempty)

    )
    (:goal (and (on yellow purple) (on purple blue) (on blue grey) (on grey green)))
)

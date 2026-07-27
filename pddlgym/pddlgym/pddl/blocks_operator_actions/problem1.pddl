(define (problem blocks)
    (:domain blocks)
    (:objects 
        blue - block
        red - block
        yellow - block
        green - block
    )
    (:init 
        (clear yellow) 
        (on yellow red) 
        (on red green)
        (on green blue)
        (ontable blue) 
        (handempty)

    )
    (:goal (and (on blue green) (on green red) (on red yellow)))
)

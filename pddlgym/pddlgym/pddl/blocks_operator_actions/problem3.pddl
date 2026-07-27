(define (problem blocks)
    (:domain blocks)
    (:objects 
        red - block
        blue - block
        grey - block
        green - block
        yellow - block
    )
    (:init 
        (clear green) 
        (clear grey) 
        (clear blue) 
        (clear red) 
        (ontable green) 
        (ontable grey)
        (ontable blue) 
        (ontable yellow)
        (on red yellow)
        (handempty)

    )
    (:goal (and (on grey blue) (on blue green) (on yellow red) (clear yellow) (clear grey)))
)







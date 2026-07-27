(define (problem blocksworld1)
    (:domain blocksworld_without_robot)
    (:objects
        green_block - block
        yellow_block - block
        red_block - block
        pink_block - block
    )
    (:init
        (ontable green_block)
        (ontable yellow_block)
        (ontable red_block)
        (ontable pink_block)
        (clear green_block)
        (clear yellow_block)
        (clear red_block)
        (clear pink_block)
        (handempty)
    )
    (:goal (and (on pink_block red_block) (on red_block yellow_block) (on yellow_block green_block)))
)
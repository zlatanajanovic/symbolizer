(define (problem blocksworld4)
    (:domain blocksworld_without_robot)
    (:objects
        orange_block - block
        yellow_block - block
        blue_block - block
        purple_block - block
        red_block - block

    )
    (:init
        (ontable orange_block)
        (ontable yellow_block)
        (ontable blue_block)
        (ontable red_block)
        (clear orange_block)
        (clear yellow_block)
        (clear blue_block)
        (clear purple_block)
        (on purple_block red_block)
        (handempty)
    )
    (:goal (and (on orange_block yellow_block) (on yellow_block blue_block) (on blue_block purple_block) (on purple_block red_block)))
)
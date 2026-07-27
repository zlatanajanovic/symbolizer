(define (problem blocksworld_real1)
  (:domain blocksworld_real)
  (:objects
    blue - block
    grey - block
    red - block
    yellow - block
  )
  (:init
    (clear yellow)
    (clear red)
    (handempty)
    (on blue grey)
    (on yellow blue)
    (ontable grey)
    (ontable red)
  )
  (:goal (and
    (on yellow blue)
    (on blue grey)
    (on grey red)
  ))
)

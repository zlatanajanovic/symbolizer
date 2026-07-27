(define (problem blocksworld_real7)
  (:domain blocksworld_real)
  (:objects
    red - block
    grey - block
    blue - block
    yellow - block
  )
  (:init
    (clear yellow)
    (handempty)
    (on grey red)
    (on blue grey)
    (on yellow blue)
    (ontable red)
  )
  (:goal (and
    (clear yellow blue)
  ))
)

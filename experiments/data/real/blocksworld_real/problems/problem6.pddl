(define (problem blocksworld_real6)
  (:domain blocksworld_real)
  (:objects
    red - block
    grey - block
    black - block
    blue - block
    white - block
    yellow - block
  )
  (:init
    (clear grey)
    (clear white)
    (clear yellow)
    (handempty)
    (on grey red)
    (on blue black)
    (on white blue)
    (ontable red)
    (ontable black)
    (ontable yellow)
  )
  (:goal (and
    (on yellow grey)
  ))
)

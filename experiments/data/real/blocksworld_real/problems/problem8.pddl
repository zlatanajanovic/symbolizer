(define (problem blocksworld_real8)
  (:domain blocksworld_real)
  (:objects
    red - block
    grey - block
    black - block
    blue - block
    yellow - block
  )
  (:init
    (clear grey)
    (clear blue)
    (clear yellow)
    (handempty)
    (on grey red)
    (on blue black)
    (ontable red)
    (ontable black)
    (ontable yellow)
  )
  (:goal (and
    (on yellow black)
  ))
)

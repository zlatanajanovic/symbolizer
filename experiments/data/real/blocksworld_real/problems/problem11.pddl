(define (problem blocksworld_real11)
  (:domain blocksworld_real)
  (:objects
    red - block
    blue - block
    white - block
  )
  (:init
    (clear blue)
    (clear white)
    (handempty)
    (on blue red)
    (ontable red)
    (ontable white)
  )
  (:goal (and
    (on white blue)
    (on blue red)
  ))
)

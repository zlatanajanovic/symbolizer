(define (problem hanoi3)
  (:domain hanoi)
  (:objects peg1 peg2 red_disk green_disk)
  (:init
   (smaller peg1 red_disk) (smaller peg1 green_disk)
   (smaller peg2 red_disk) (smaller peg2 green_disk)
   (smaller green_disk red_disk)
   (clear red_disk) (clear green_disk)
   (on red_disk peg1) (on green_disk peg2)
   
   
   
   
   
   
  )
  (:goal (and (on red_disk green_disk)))
  )
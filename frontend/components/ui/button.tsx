import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap text-sm font-medium font-mono uppercase tracking-wider transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default:
          "bg-transparent border-2 border-accent text-accent cyber-chamfer hover:bg-accent hover:text-accent-foreground hover:shadow-neon focus-visible:shadow-neon-sm",
        destructive:
          "bg-destructive text-destructive-foreground hover:bg-destructive/90 rounded-sm border border-destructive",
        outline:
          "border border-border bg-transparent rounded-sm hover:border-accent hover:text-accent hover:shadow-neon-sm",
        secondary:
          "bg-transparent border-2 border-secondary text-secondary cyber-chamfer hover:bg-secondary hover:text-secondary-foreground hover:shadow-neon-secondary focus-visible:shadow-neon-secondary-sm",
        ghost:
          "border-0 rounded-sm hover:bg-muted hover:text-foreground",
        link:
          "text-accent underline-offset-4 hover:underline rounded-sm",
        glitch:
          "bg-accent text-accent-foreground border-2 border-accent cyber-chamfer cyber-glitch hover:shadow-neon focus-visible:shadow-neon",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-sm px-3",
        lg: "h-11 rounded-sm px-8",
        icon: "h-10 w-10 rounded-sm",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => {
    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }

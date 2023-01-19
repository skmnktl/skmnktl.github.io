# Quicksort

```rust

pub fn randomize_pivot(a: &mut Vec<i32>, p: i32, r: i32) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(p..r);
    a.swap(r as usize, i as usize);
}

pub fn quick_sort(a: &mut Vec<i32>, p: i32, r: i32) {
    if p < r {
        let q = partition(a, p, r);
        quick_sort(a, p, q - 1);
        quick_sort(a, q + 1, r);
    }
}

pub fn partition(a: &mut Vec<i32>, p: i32, r: i32) -> i32 {
    randomize_pivot(a, p, r);
    let pivot = a[r as usize];
    let mut i = p-1;
    for j in p..r{
        if a[j as usize] <= pivot{
            i+=1;
            a.swap(i as usize, j as usize);
        }
    }
    a.swap( (i+1) as usize , r as usize);
    return i+1
}
```

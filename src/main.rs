extern crate find_folder;
extern crate piston_window;

use piston_window::*;
use tch::jit::CModule;
use tch::Kind;
use tch::Tensor;


const POINT_SIZE: f64 = 21.0;

fn main() {
    let mut window: PistonWindow = WindowSettings::new("CharRec", [830, 630])
        .exit_on_esc(true)
        .resizable(false)
        .samples(4)
        .build()
        .unwrap();

    window.set_lazy(true);

    let assets = find_folder::Search::ParentsThenKids(3, 3)
        .for_folder("assets")
        .unwrap();
    let mut glyphs = window
        .load_font(assets.join("FiraSans-Regular.ttf"))
        .unwrap();

    let model = match CModule::load(assets.join("mnist_cnn.pth")) {
        Ok(x) => x,
        Err(e) => panic!("{}", e),
    };

    let digits: [&str; 10] = [&"0", &"1", &"2", &"3", &"4", &"5", &"6", &"7", &"8", &"9"];

    let mut cursor = [0.0, 0.0];
    let mut is_drawing = false;

    let mut state = State::new();
    let button = MyButton::new([650.0, 300.0, 100.0, 50.0]);

    let mut guess = 0;
    let brush = [[0.1, 0.25, 0.1], [0.25, 1.0, 0.25], [0.1, 0.25, 0.1]];

    while let Some(event) = window.next() {
        event.mouse_cursor(|pos| cursor = pos);

        if let Some(Button::Mouse(_button)) = event.press_args() {
            is_drawing = true;
        }
        if let Some(Button::Mouse(_button)) = event.release_args() {
            is_drawing = false;
            let mut data = [0.0f32; 784];
            let mut i = 0;
            for x in 0..28 {
                for y in 0..28 {
                    data[i] = state.points[y][x];
                    i += 1;
                }
            }
            // SAFETY: `data` has the same lifetime as `m_data` and `data` is never modified
            let m_data: *const [f32] = &data;
            let input = Tensor::of_data_size(
                unsafe { &*(m_data as *const [u8]) },
                &[1, 1, 28, 28],
                Kind::Float,
            );
            input.print();
            let output = model
                .forward_ts(&[input])
                .unwrap()
                .softmax(-1, Kind::Float)
                .get(0);
            output.print();
            let mut largest = 0.0;
            for i in 0..10 {
                let num = f32::from(output.get(i));
                if num > largest {
                    largest = num;
                    guess = i as usize;
                }
            }
            println!("{}", guess);
        }

        if is_drawing {
            let x = (cursor[0] / POINT_SIZE) as usize;
            let y = (cursor[1] / POINT_SIZE) as usize;
            if x < 28 && y < 28 {
                state.points[x][y] = 1.0;
                let lx = 0;
                let x_start = if x == 0 { x } else { x - 1 };
                for m_x in x_start..(x + 2) {
                    if m_x < 28 {
                        let ly = 0;
                        let y_start = if y == 0 { y } else { y - 1 };
                        for m_y in y_start..(y + 2) {
                            if m_y < 28 {
                                state.points[m_x][m_y] += brush[lx][ly];
                                if state.points[m_x][m_y] > 1.0 {
                                    state.points[m_x][m_y] = 1.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        window.draw_2d(&event, |mut c, mut g, dev| {
            clear([0.8, 0.8, 0.8, 1.0], g);

            rectangle(
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 28.0 * POINT_SIZE, 28.0 * POINT_SIZE],
                c.transform,
                g,
            );
            for x in 0..28 {
                for y in 0..28 {
                    if state.points[x][y] != 0.0 {
                        let clr = state.points[x][y];
                        rectangle(
                            [1.0 - clr, 1.0 - clr, 1.0 - clr, 1.0],
                            [
                                x as f64 * POINT_SIZE,
                                y as f64 * POINT_SIZE,
                                POINT_SIZE,
                                POINT_SIZE,
                            ],
                            c.transform,
                            g,
                        );
                    }
                }
            }
            button.draw(
                &mut state.points,
                |points: &mut [[f32; 28]; 28]| {
                    for x in 0..28 {
                        for y in 0..28 {
                            points[x][y] = 0.0;
                        }
                    }
                },
                &cursor,
                &is_drawing,
                &mut glyphs,
                &mut c,
                &mut g,
            );
            text::Text::new_color([0.0, 1.0, 0.0, 1.0], 300)
                .draw(
                    digits[guess],
                    &mut glyphs,
                    &c.draw_state,
                    c.transform.trans(620.0, 270.0),
                    g,
                )
                .unwrap();
            glyphs.factory.encoder.flush(dev);
        });
    }
}

struct State {
    points: [[f32; 28]; 28],
}
impl State {
    fn new() -> Self {
        Self {
            points: [[0f32; 28]; 28],
        }
    }
}

struct MyButton {
    dims: [f64; 4],
}
impl MyButton {
    fn new(dims: [f64; 4]) -> Self {
        Self { dims }
    }
    fn draw<F>(
        &self,
        points: &mut [[f32; 28]; 28],
        cb: F,
        cursor: &[f64; 2],
        is_clicked: &bool,
        glyphs: &mut Glyphs,
        c: &mut Context,
        g: &mut G2d,
    ) where
        F: FnOnce(&mut [[f32; 28]; 28]),
    {
        let mut color = [1.0, 1.0, 1.0, 1.0];
        if cursor[0] > self.dims[0]
            && cursor[0] < self.dims[0] + self.dims[2]
            && cursor[1] > self.dims[1]
            && cursor[1] < self.dims[1] + self.dims[3]
        {
            if *is_clicked {
                color = [0.5, 0.5, 0.5, 1.0];
                cb(points);
            } else {
                color = [0.7, 0.7, 0.7, 1.0];
            }
        }
        rectangle(color, self.dims, c.transform, g);
        text::Text::new_color([1.0, 0.0, 0.0, 1.0], 32)
            .draw(
                "Clear",
                glyphs,
                &c.draw_state,
                c.transform.trans(self.dims[0] + 13.0, self.dims[1] + 34.0),
                g,
            )
            .unwrap();
    }
}
